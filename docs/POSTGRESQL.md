# PostgreSQL storage design

The reference schema is in `migrations/0001_core.sql`. PostgreSQL 16+ and pgvector
0.7+ are the target. Application repositories must always bind `tenant_id` and
`project_id`; row-level security should be enabled by the deployment-specific
migration after the application's database roles are established.

## Data boundaries

- Relational columns hold identifiers, tenant keys, status, timestamps, sequence,
  versions, and fields used in filters or joins.
- JSONB holds provider-neutral typed payloads and state snapshots whose internal
  keys vary by application.
- Persisted Pydantic payloads include a `schema_version` wrapper before adapters
  write them.
- Vector embeddings live beside their authoritative chunk text and provenance.
- Audit rows are append-only. The application database role receives no update or
  delete privilege on them.

## Consistency

- Execution events use `(tenant_id, project_id, execution_id, sequence)` as their
  uniqueness boundary.
- Checkpoints use optimistic revision and an idempotency key.
- An execution state update and its checkpoint are written in the same transaction.
- Tool/model side effects use a stable node-attempt idempotency key outside this
  database when the remote provider supports one.
- Document chunk replacement occurs in one transaction: insert the new document
  version/chunks, switch the current version, then retire older chunks.

## Indexing

- B-tree indexes serve tenant/time, execution sequence, status, and expiry scans.
- GIN indexes serve event/state metadata and sparse `tsvector` retrieval.
- HNSW cosine indexes serve pgvector dense retrieval. Build parameters must be
  benchmarked with the production embedding dimension and corpus.
- Partial indexes exclude expired memory from default lookup.

## Scaling and retention

- Partition high-volume `execution_events`, `audit_logs`, and `usage_records` by
  month once a deployment exceeds roughly tens of millions of rows. Create future
  partitions before the month boundary and detach/archive old partitions.
- Keep checkpoints for active/resumable executions; compact completed execution
  checkpoints according to organization policy.
- Move immutable event/audit archives to encrypted object storage with a manifest
  and checksum before deletion.
- Use read replicas for dashboards and evaluation queries, never runtime writes.
- PgBouncer transaction pooling is supported as long as adapters do not depend on
  session state.

## Backups and recovery

- Enable encrypted daily base backups plus continuous WAL archiving.
- Test point-in-time restoration at least quarterly in an isolated account.
- Back up encryption metadata and external secret-provider configuration through
  their respective supported mechanisms.
- A restore is incomplete until event sequence constraints, checkpoint resume, and
  vector/keyword retrieval smoke tests pass.

## Migration policy

- Migrations are ordered, immutable after release, and forward-only in production.
- Expand/contract changes span at least one compatible release: add nullable/new
  representation, dual read/write, backfill, switch reads, then remove old data.
- Destructive migrations require an export/restore procedure and explicit operator
  confirmation.
- Workers check the database schema compatibility range before accepting runs.

