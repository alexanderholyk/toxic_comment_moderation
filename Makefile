# Makefile for local dev (Docker)
SHELL := /bin/bash

COMPOSE := docker compose -f infra/docker/docker-compose.dev.yml
UI_URL := http://localhost:8501
MONITOR_URL := http://localhost:8502
API_HEALTH := http://127.0.0.1:8000/health

# Choose the right "open a browser" command
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	OPEN := xdg-open
else
	OPEN := open
endif

.PHONY: help build run stop down clean logs ps restart open wait

help:
	@echo "Targets:"
	@echo "  make build    - Build all images"
	@echo "  make run      - Up (detached), wait for API, open UI & Monitoring"
	@echo "  make stop     - Stop containers (keep state)"
	@echo "  make down     - Stop & remove containers/network"
	@echo "  make clean    - Down + remove anonymous volumes & dangling images"
	@echo "  make logs     - Follow logs"
	@echo "  make ps       - Show container status"
	@echo "  make restart  - Recreate containers"
	@echo "  make open     - Open UI and Monitoring in browser"

build:
	$(COMPOSE) build --pull

run:
	$(COMPOSE) up -d --build
	@$(MAKE) wait
	@$(MAKE) open
	@$(MAKE) ps

stop:
	$(COMPOSE) stop

down:
	$(COMPOSE) down

clean:
	$(COMPOSE) down -v
	@docker image prune -f >/dev/null

logs:
	$(COMPOSE) logs -f --tail=200

ps:
	$(COMPOSE) ps

restart:
	$(COMPOSE) up -d --force-recreate
	@$(MAKE) wait
	@$(MAKE) open
	@$(MAKE) ps

open:
	@echo "Opening dashboards…"
	@$(OPEN) "$(UI_URL)" >/dev/null 2>&1 || true
	@$(OPEN) "$(MONITOR_URL)" >/dev/null 2>&1 || true

# Wait until the API health endpoint responds (up to ~60s)
wait:
	@echo "Waiting for API: $(API_HEALTH)"
	@i=0; \
	until curl -fsS "$(API_HEALTH)" >/dev/null || [ $$i -ge 60 ]; do \
		i=$$((i+1)); sleep 1; \
	done; \
	if [ $$i -ge 60 ]; then \
		echo "Timed out waiting for API."; exit 1; \
	else \
		echo "API is up!"; \
	fi