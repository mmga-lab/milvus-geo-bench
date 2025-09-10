# Makefile for Milvus Geo Benchmark Tool

.PHONY: help format check install clean dev-install

# Default target
help:
	@echo "Available commands:"
	@echo "  format      - Format code using ruff"
	@echo "  check       - Check code quality using ruff"
	@echo "  install     - Install package dependencies"
	@echo "  dev-install - Install package with dev dependencies"
	@echo "  clean       - Clean generated files"

# Code formatting
format:
	@echo "🔧 Formatting code with ruff..."
	uv run ruff format src/
	@echo "✅ Code formatting completed!"

# Code quality check
check:
	@echo "🔍 Checking code quality with ruff..."
	uv run ruff check src/
	@echo "✅ Code quality check completed!"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	uv sync
	@echo "✅ Dependencies installed!"

# Install with dev dependencies
dev-install:
	@echo "📦 Installing with dev dependencies..."
	uv sync --group dev
	@echo "✅ Dev dependencies installed!"

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf data/
	rm -rf reports/
	rm -rf .ruff_cache/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "✅ Cleanup completed!"