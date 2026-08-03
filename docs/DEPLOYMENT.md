# Deployment

## Local container stack

```bash
docker compose -f deploy/docker-compose.yml up --build
curl http://localhost:8000/health
```

The included server runs the deterministic echo graph. Replace
`examples.server:application` with your own `Application` import target. The
PostgreSQL and Redis services establish the intended production topology; v0.1
uses in-memory stores until their adapters are configured in a later milestone.

Never use the Compose password outside local development.

## Production topology

- Place the API behind TLS termination and an authenticated gateway.
- Run API and durable workers as separate deployments once the distributed queue
  adapter is introduced.
- Use managed PostgreSQL with pgvector, point-in-time recovery, and private network
  access.
- Use a managed Redis-compatible service for queues/rate limits, not as the source
  of truth for executions.
- Inject secrets through a cloud secret manager or CSI driver; never bake them into
  an image or manifest.
- Export logs, metrics, and traces out of process.

The Kubernetes manifest is a secure baseline, not a complete production release.
Replace the example image, add NetworkPolicy/PodDisruptionBudget/Ingress, and tune
resources/probes using load tests.

## Shutdown order

1. Mark readiness false so the load balancer stops new requests.
2. Stop admitting new executions.
3. Cancel or checkpoint active resumable work.
4. Flush bounded telemetry queues without blocking indefinitely.
5. Close provider/database clients and exit before the grace period.

The single-process v0.1 server relies on ASGI lifecycle cancellation. A durable
worker drain controller is planned with the distributed runtime.

## Cloud notes

- AWS: ECS/Fargate or EKS, RDS PostgreSQL, ElastiCache, Secrets Manager, and ADOT.
- Azure: Container Apps or AKS, Flexible Server for PostgreSQL, Managed Redis,
  Key Vault, and Azure Monitor OpenTelemetry.
- GCP: Cloud Run or GKE, Cloud SQL PostgreSQL, Memorystore, Secret Manager, and
  Cloud Trace/Monitoring.
- Fly.io, Railway, Render, and DigitalOcean can run the same container; use their
  managed PostgreSQL, private networking, health checks, and secret stores.

