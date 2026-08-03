"""Budgeted multi-agent team primitives built on ordinary agent runs."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import Field, model_validator

from praxium.agents import Agent, AgentResult, AgentRunner
from praxium.core import FrameworkModel, Message, Usage


class TeamRole(StrEnum):
    SUPERVISOR = "supervisor"
    PLANNER = "planner"
    RESEARCHER = "researcher"
    WRITER = "writer"
    REVIEWER = "reviewer"
    CRITIC = "critic"
    EXECUTOR = "executor"
    WORKER = "worker"


class TeamMember(FrameworkModel):
    name: str = Field(min_length=1)
    role: TeamRole
    agent: Agent
    shares_output: bool = True


class TeamPolicy(FrameworkModel):
    max_rounds: int = Field(default=1, ge=1, le=100)
    max_delegations: int = Field(default=20, ge=1)
    max_depth: int = Field(default=3, ge=1)
    stop_on_error: bool = True


class Team(FrameworkModel):
    name: str
    members: list[TeamMember] = Field(min_length=1)
    policy: TeamPolicy = Field(default_factory=TeamPolicy)

    @model_validator(mode="after")
    def validate_members(self) -> Team:
        names = [member.name for member in self.members]
        if len(names) != len(set(names)):
            raise ValueError("team member names must be unique")
        return self


class Delegation(FrameworkModel):
    member: str
    instruction: str
    depth: int = Field(default=1, ge=1)


class TeamTurn(FrameworkModel):
    round: int = Field(ge=1)
    member: str
    role: TeamRole
    instruction: str
    response: Message
    usage: Usage


class TeamResult(FrameworkModel):
    team: str
    turns: list[TeamTurn]
    final_response: Message
    usage: Usage


class DelegationStrategy(Protocol):
    async def plan(
        self,
        team: Team,
        prompt: str,
        turns: list[TeamTurn],
        round_number: int,
    ) -> list[Delegation]: ...


class RoundRobinStrategy(FrameworkModel):
    """Deterministic strategy that visits each member once per round."""

    instruction_template: str = "{prompt}\n\nPrevious contribution:\n{previous}"

    async def plan(
        self,
        team: Team,
        prompt: str,
        turns: list[TeamTurn],
        round_number: int,
    ) -> list[Delegation]:
        del round_number
        previous = turns[-1].response.text_content if turns else "(none)"
        return [
            Delegation(
                member=member.name,
                instruction=self.instruction_template.format(prompt=prompt, previous=previous),
            )
            for member in team.members
        ]


class TeamRunner:
    """Executes explicit delegation plans with hard recursion and work budgets."""

    def __init__(
        self, agent_runner: AgentRunner, strategy: DelegationStrategy | None = None
    ) -> None:
        self.agent_runner = agent_runner
        self.strategy = strategy or RoundRobinStrategy()

    async def run(self, team: Team, prompt: str) -> TeamResult:
        member_map = {member.name: member for member in team.members}
        turns: list[TeamTurn] = []
        usage = Usage()
        delegations = 0
        for round_number in range(1, team.policy.max_rounds + 1):
            plan = await self.strategy.plan(team, prompt, list(turns), round_number)
            for delegation in plan:
                delegations += 1
                if delegations > team.policy.max_delegations:
                    raise RuntimeError("team exceeded its delegation budget")
                if delegation.depth > team.policy.max_depth:
                    raise RuntimeError("delegation exceeded the team's recursion depth")
                member = member_map.get(delegation.member)
                if member is None:
                    raise ValueError(f"strategy selected unknown team member {delegation.member!r}")
                result: AgentResult = await self.agent_runner.run(
                    member.agent, delegation.instruction
                )
                usage = usage + result.usage
                turns.append(
                    TeamTurn(
                        round=round_number,
                        member=member.name,
                        role=member.role,
                        instruction=delegation.instruction,
                        response=result.response,
                        usage=result.usage,
                    )
                )
        if not turns:
            raise RuntimeError("delegation strategy produced no work")
        return TeamResult(
            team=team.name,
            turns=turns,
            final_response=turns[-1].response,
            usage=usage,
        )
