COMPOSE ?= docker compose

.PHONY: help up up-db down clean logs ps health wait test lint fmt evals

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

up:  ## Start everything (postgres + Langfuse)
	$(COMPOSE) up -d

up-db:  ## Start only postgres (enough for tests and indexing)
	$(COMPOSE) up -d postgres

down:  ## Stop all services, keep data
	$(COMPOSE) down

clean:  ## Stop and DELETE all volumes (re-runs the postgres init script)
	$(COMPOSE) down -v

ps:  ## Show service status
	$(COMPOSE) ps

logs:  ## Tail logs
	$(COMPOSE) logs -f

wait:  ## Block until postgres is healthy
	@until [ "$$($(COMPOSE) ps --format '{{.Health}}' postgres)" = "healthy" ]; do \
	  echo "waiting for postgres..."; sleep 2; \
	done; echo "postgres healthy"

health:  ## Verify Phase 2: schema present, pgvector enabled, Langfuse reachable
	@echo "--- extensions ---"
	@$(COMPOSE) exec -T postgres psql -U $${POSTGRES_USER:-postgres} -d $${POSTGRES_DB:-ragbot} \
	  -c "SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector','pgcrypto');"
	@echo "--- tables ---"
	@$(COMPOSE) exec -T postgres psql -U $${POSTGRES_USER:-postgres} -d $${POSTGRES_DB:-ragbot} \
	  -c "\dt"
	@echo "--- embedding column ---"
	@$(COMPOSE) exec -T postgres psql -U $${POSTGRES_USER:-postgres} -d $${POSTGRES_DB:-ragbot} \
	  -c "SELECT atttypmod AS dim FROM pg_attribute WHERE attrelid='chunks'::regclass AND attname='embedding';"
	@echo "--- databases ---"
	@$(COMPOSE) exec -T postgres psql -U $${POSTGRES_USER:-postgres} -d postgres \
	  -c "SELECT datname FROM pg_database WHERE datname IN ('$${POSTGRES_DB:-ragbot}','$${LANGFUSE_DB:-langfuse}');"
	@echo "--- langfuse ---"
	@curl -fsS -o /dev/null -w "langfuse http %{http_code}\n" http://localhost:3000 || echo "langfuse NOT reachable"

test:  ## Run the test suite
	.venv/bin/python -m pytest -q

lint:  ## Lint and format check
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

evals:  ## Score the golden dataset with RAGAS (needs postgres + a model)
	.venv/bin/python -m evals.run_ragas --min-faithfulness 0.85 --min-context-recall 0.60

fmt:  ## Auto-format
	.venv/bin/ruff format . && .venv/bin/ruff check --fix .
