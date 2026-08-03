BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE organizations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    slug text NOT NULL UNIQUE,
    name text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE projects (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id uuid NOT NULL REFERENCES organizations(id),
    slug text NOT NULL,
    name text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (organization_id, slug),
    UNIQUE (organization_id, id)
);

CREATE TABLE conversations (
    tenant_id uuid NOT NULL REFERENCES organizations(id),
    project_id uuid NOT NULL,
    id text NOT NULL,
    schema_version integer NOT NULL DEFAULT 1,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, project_id, id),
    FOREIGN KEY (tenant_id, project_id) REFERENCES projects(organization_id, id)
);

CREATE TABLE messages (
    tenant_id uuid NOT NULL,
    project_id uuid NOT NULL,
    conversation_id text NOT NULL,
    id text NOT NULL,
    sequence bigint NOT NULL,
    role text NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    payload jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, project_id, conversation_id, id),
    UNIQUE (tenant_id, project_id, conversation_id, sequence),
    FOREIGN KEY (tenant_id, project_id, conversation_id)
        REFERENCES conversations(tenant_id, project_id, id) ON DELETE CASCADE
);

CREATE TABLE executions (
    tenant_id uuid NOT NULL,
    project_id uuid NOT NULL,
    id text NOT NULL,
    graph_id text NOT NULL,
    graph_version integer NOT NULL,
    graph_fingerprint text NOT NULL,
    status text NOT NULL CHECK (status IN (
        'pending', 'running', 'completed', 'failed', 'cancelled', 'timed_out', 'suspended'
    )),
    state_version bigint NOT NULL DEFAULT 0,
    state jsonb NOT NULL DEFAULT '{}'::jsonb,
    output jsonb,
    error jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    PRIMARY KEY (tenant_id, project_id, id),
    FOREIGN KEY (tenant_id, project_id) REFERENCES projects(organization_id, id)
);

CREATE INDEX executions_status_updated_idx
    ON executions (tenant_id, project_id, status, updated_at DESC);
CREATE INDEX executions_metadata_gin_idx ON executions USING gin (metadata);

CREATE TABLE execution_events (
    tenant_id uuid NOT NULL,
    project_id uuid NOT NULL,
    execution_id text NOT NULL,
    event_id text NOT NULL,
    sequence bigint NOT NULL CHECK (sequence > 0),
    kind text NOT NULL,
    node_id text,
    attempt integer,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, project_id, execution_id, sequence),
    UNIQUE (tenant_id, project_id, event_id),
    FOREIGN KEY (tenant_id, project_id, execution_id)
        REFERENCES executions(tenant_id, project_id, id) ON DELETE CASCADE
);

CREATE INDEX execution_events_kind_time_idx
    ON execution_events (tenant_id, project_id, kind, created_at DESC);
CREATE INDEX execution_events_payload_gin_idx ON execution_events USING gin (payload);

CREATE TABLE checkpoints (
    tenant_id uuid NOT NULL,
    project_id uuid NOT NULL,
    id text NOT NULL,
    execution_id text NOT NULL,
    revision bigint NOT NULL DEFAULT 1 CHECK (revision > 0),
    graph_version integer NOT NULL,
    graph_fingerprint text NOT NULL,
    state_version bigint NOT NULL,
    state jsonb NOT NULL,
    next_node text,
    visit_counts jsonb NOT NULL DEFAULT '{}'::jsonb,
    completed_nodes jsonb NOT NULL DEFAULT '[]'::jsonb,
    output jsonb,
    idempotency_key text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, project_id, id),
    UNIQUE (tenant_id, project_id, execution_id, revision),
    UNIQUE (tenant_id, project_id, idempotency_key),
    FOREIGN KEY (tenant_id, project_id, execution_id)
        REFERENCES executions(tenant_id, project_id, id) ON DELETE CASCADE
);

CREATE INDEX checkpoints_execution_created_idx
    ON checkpoints (tenant_id, project_id, execution_id, created_at DESC);

CREATE TABLE memory_records (
    tenant_id uuid NOT NULL,
    project_id uuid NOT NULL,
    namespace text NOT NULL,
    id text NOT NULL,
    version integer NOT NULL CHECK (version > 0),
    kind text NOT NULL,
    branch text NOT NULL DEFAULT 'main',
    parent_id text,
    content text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding vector(1536),
    expires_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, project_id, namespace, id, version),
    FOREIGN KEY (tenant_id, project_id) REFERENCES projects(organization_id, id)
);

CREATE INDEX memory_current_lookup_idx
    ON memory_records (tenant_id, project_id, namespace, branch, id, version DESC);
CREATE INDEX memory_expiration_idx
    ON memory_records (tenant_id, project_id, expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX memory_metadata_gin_idx ON memory_records USING gin (metadata);
CREATE INDEX memory_search_gin_idx ON memory_records USING gin (search_vector);
CREATE INDEX memory_embedding_hnsw_idx
    ON memory_records USING hnsw (embedding vector_cosine_ops) WHERE embedding IS NOT NULL;

CREATE TABLE documents (
    tenant_id uuid NOT NULL,
    project_id uuid NOT NULL,
    namespace text NOT NULL,
    collection text NOT NULL,
    id text NOT NULL,
    version integer NOT NULL CHECK (version > 0),
    source_uri text,
    content_checksum text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, project_id, namespace, collection, id, version),
    FOREIGN KEY (tenant_id, project_id) REFERENCES projects(organization_id, id)
);

CREATE TABLE document_chunks (
    tenant_id uuid NOT NULL,
    project_id uuid NOT NULL,
    namespace text NOT NULL,
    collection text NOT NULL,
    document_id text NOT NULL,
    document_version integer NOT NULL,
    id text NOT NULL,
    chunk_index integer NOT NULL CHECK (chunk_index >= 0),
    start_offset integer NOT NULL CHECK (start_offset >= 0),
    end_offset integer NOT NULL CHECK (end_offset >= start_offset),
    content text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding vector(1536),
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, project_id, namespace, collection, id),
    FOREIGN KEY (tenant_id, project_id, namespace, collection, document_id, document_version)
        REFERENCES documents(tenant_id, project_id, namespace, collection, id, version)
        ON DELETE CASCADE
);

CREATE INDEX document_chunks_search_gin_idx ON document_chunks USING gin (search_vector);
CREATE INDEX document_chunks_metadata_gin_idx ON document_chunks USING gin (metadata);
CREATE INDEX document_chunks_embedding_hnsw_idx
    ON document_chunks USING hnsw (embedding vector_cosine_ops) WHERE embedding IS NOT NULL;

CREATE TABLE audit_logs (
    tenant_id uuid NOT NULL,
    project_id uuid NOT NULL,
    id text NOT NULL,
    principal_id text NOT NULL,
    action text NOT NULL,
    resource_type text NOT NULL,
    resource_id text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, project_id, id),
    FOREIGN KEY (tenant_id, project_id) REFERENCES projects(organization_id, id)
);

CREATE INDEX audit_logs_resource_time_idx
    ON audit_logs (tenant_id, project_id, resource_type, resource_id, created_at DESC);
CREATE INDEX audit_logs_principal_time_idx
    ON audit_logs (tenant_id, project_id, principal_id, created_at DESC);

COMMIT;
