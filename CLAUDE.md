# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment Setup

This project uses `uv` as the package manager and requires Python 3.12+. The project dependencies include:
- `pymilvus` (installed from TestPyPI) for Milvus database operations
- `shapely` for geospatial computations and ground truth calculations
- `pandas` and `pyarrow` for data processing with Parquet format
- `click` for the CLI interface

### Essential Commands

```bash
# Install dependencies
make install

# Install with development tools
make dev-install

# Format code (uses ruff)
make format

# Check code quality (uses ruff)
make check

# Clean generated files
make clean

# Run the tool
milvus-geo-bench --help
```

### Key Configuration Files

- `config.yaml.example` - Template for runtime configuration (copy to `config.yaml`)
- `.env.example` - Template for environment variables (copy to `.env`)
- Set `MILVUS_URI` and `MILVUS_TOKEN` environment variables for Milvus connection

## Architecture Overview

This is a comprehensive benchmarking tool for Milvus geo search functionality with a modular CLI-based architecture:

### Data Flow Pipeline
1. **Dataset Generation** (`dataset.py`) → Creates synthetic geospatial data
2. **Data Loading** (`milvus_client.py`) → Inserts data into Milvus collections  
3. **Benchmark Execution** (`benchmark.py`) → Runs performance tests
4. **Evaluation** (`evaluator.py`) → Compares results against ground truth

### Core Modules

**`dataset.py` - DatasetGenerator**
- Generates random geospatial points within bounding boxes
- Creates hexagonal polygon queries that guarantee minimum result counts
- Uses Shapely to calculate ground truth for spatial queries
- All data saved in Parquet format for performance

**`milvus_client.py` - MilvusGeoClient**
- Wraps Milvus operations with URI/token authentication
- Creates collections with GEOMETRY fields for spatial data
- Handles batch data insertion and query execution
- Includes automatic indexing (vector + geometry indexes)

**`benchmark.py` - Benchmark/BenchmarkRunner**
- Executes performance tests with warmup queries
- Measures query latency and throughput
- Records success/failure rates and error details
- Supports configurable timeouts and concurrency

**`evaluator.py` - Evaluator**
- Calculates precision, recall, F1-score metrics
- Generates detailed markdown and JSON reports
- Provides both macro and micro averaging
- Includes performance statistics and distribution analysis

**`utils.py`**
- Configuration management with environment variable substitution
- Parquet I/O operations
- Logging setup and utility functions

### CLI Commands Structure

The tool provides 5 main commands accessible via `milvus-geo-bench`:

- `generate-dataset` - Create synthetic training/test data
- `load-data` - Insert data into Milvus collections
- `run-benchmark` - Execute performance tests
- `evaluate` - Compare results against ground truth  
- `full-run` - Complete end-to-end workflow

Each command accepts both CLI parameters and configuration files, with CLI args taking precedence.

### Data Formats

All datasets use Parquet format:
- **train.parquet**: `id`, `wkt` (Point), `vec` (8D vector)
- **test.parquet**: `query_id`, `expr` (ST_WITHIN), `polygon_wkt`, center coordinates, radius
- **ground_truth.parquet**: `query_id`, `result_ids`, `result_count`
- **results.parquet**: `query_id`, `query_time_ms`, `result_ids`, `result_count`, `success`

### Spatial Query Strategy

- Uses ST_WITHIN queries with hexagonal polygons
- Dynamically adjusts radius to ensure minimum result counts (default: 100 points)
- Ground truth calculated using Shapely's geometric operations
- Supports configurable bounding boxes for different geographic regions

### Configuration System

Multi-layer configuration with precedence:
1. CLI arguments (highest)
2. Configuration files (config.yaml)
3. Environment variables (${VAR_NAME} substitution)  
4. Default values (lowest)

Milvus connection requires URI/token authentication, typically configured via environment variables.