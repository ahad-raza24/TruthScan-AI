# TruthScan - Simple Makefile

# Variables
APP_NAME := truthscan

# Colors
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

.DEFAULT_GOAL := help

# Help
.PHONY: help
help: ## Show available commands
	@echo "$(GREEN)TruthScan - Available Commands$(NC)"
	@echo "=============================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Setup
.PHONY: setup
setup: ## Set up development environment
	@echo "$(GREEN)Setting up development environment...$(NC)"
	@if [ ! -f .env ]; then cp .env.example .env; echo "$(YELLOW)Created .env file - please add your API keys!$(NC)"; fi
	@python3 -m venv venv
	@. venv/bin/activate && pip install -r requirements.txt
	@echo "$(GREEN)Setup complete! Run: source venv/bin/activate$(NC)"

# Local development
.PHONY: dev
dev: ## Start development server locally
	@echo "$(GREEN)Starting TruthScan locally...$(NC)"
	@python app.py

# Docker commands
.PHONY: build
build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	@docker build -t $(APP_NAME) .

.PHONY: run
run: ## Run with Docker
	@echo "$(GREEN)Starting TruthScan with Docker...$(NC)"
	@docker run -d --name $(APP_NAME) -p 5001:5001 --env-file .env $(APP_NAME)

.PHONY: up
up: ## Start with docker-compose
	@echo "$(GREEN)Starting TruthScan with docker-compose...$(NC)"
	@docker-compose up -d
	@echo "$(GREEN)Access at: http://localhost:5001$(NC)"

.PHONY: down
down: ## Stop docker-compose services
	@echo "$(YELLOW)Stopping services...$(NC)"
	@docker-compose down

.PHONY: logs
logs: ## Show application logs
	@docker-compose logs -f truthscan

# Cleanup
.PHONY: clean
clean: ## Clean up Docker resources
	@echo "$(YELLOW)Cleaning up...$(NC)"
	@docker-compose down -v
	@docker system prune -f

# Quick start
.PHONY: quick-start
quick-start: build up ## Build and start everything
	@echo "$(GREEN)TruthScan is running at http://localhost:5001$(NC)"
	@echo "$(YELLOW)Remember to add your API keys to .env file!$(NC)" 