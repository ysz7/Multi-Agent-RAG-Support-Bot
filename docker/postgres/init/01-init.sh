#!/bin/bash
# Runs once, on first boot of an empty postgres volume.
#
#   1. creates the Langfuse database alongside the app database
#   2. enables pgvector in the app database
#   3. creates the documents / chunks tables, sized from EMBEDDING_DIM
#
# To re-run after changing EMBEDDING_DIM: docker compose down -v && docker compose up -d
set -euo pipefail

APP_DB="${POSTGRES_DB:-ragbot}"
LANGFUSE_DB="${LANGFUSE_DB:-langfuse}"
EMBEDDING_DIM="${EMBEDDING_DIM:-768}"

if ! [[ "$EMBEDDING_DIM" =~ ^[0-9]+$ ]]; then
  echo "EMBEDDING_DIM must be an integer, got: $EMBEDDING_DIM" >&2
  exit 1
fi

echo "init: creating database '$LANGFUSE_DB'"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<-EOSQL
	CREATE DATABASE "$LANGFUSE_DB";
EOSQL

echo "init: preparing '$APP_DB' with vector($EMBEDDING_DIM)"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$APP_DB" <<-EOSQL
	CREATE EXTENSION IF NOT EXISTS vector;
	CREATE EXTENSION IF NOT EXISTS pgcrypto;

	-- One row per ingested source file.
	CREATE TABLE IF NOT EXISTS documents (
	    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
	    tenant_id     text        NOT NULL,
	    source_path   text        NOT NULL,
	    title         text,
	    content_hash  text        NOT NULL,
	    metadata      jsonb       NOT NULL DEFAULT '{}'::jsonb,
	    created_at    timestamptz NOT NULL DEFAULT now(),
	    updated_at    timestamptz NOT NULL DEFAULT now(),
	    -- Makes re-indexing idempotent: same tenant + same bytes = same row.
	    CONSTRAINT documents_tenant_hash_key UNIQUE (tenant_id, content_hash)
	);

	CREATE INDEX IF NOT EXISTS documents_tenant_idx ON documents (tenant_id);

	-- One row per embedded chunk.
	CREATE TABLE IF NOT EXISTS chunks (
	    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
	    document_id  uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
	    tenant_id    text NOT NULL,
	    chunk_index  int  NOT NULL,
	    content      text NOT NULL,
	    metadata     jsonb NOT NULL DEFAULT '{}'::jsonb,
	    embedding    vector($EMBEDDING_DIM),
	    created_at   timestamptz NOT NULL DEFAULT now(),
	    CONSTRAINT chunks_document_index_key UNIQUE (document_id, chunk_index)
	);

	-- Tenant filters are applied server-side on every query (see PLAN.md).
	CREATE INDEX IF NOT EXISTS chunks_tenant_idx ON chunks (tenant_id);

	-- HNSW + cosine distance; matches the operator the retriever uses (<=>).
	CREATE INDEX IF NOT EXISTS chunks_embedding_idx
	    ON chunks USING hnsw (embedding vector_cosine_ops);
EOSQL

echo "init: done"
