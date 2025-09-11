# Milvus Geo Search Benchmark Tool

A comprehensive benchmark tool for evaluating Milvus geo search functionality with synthetic datasets.

## Features

- **Dataset Generation**: Create synthetic geospatial datasets with points and polygons
- **Ground Truth Calculation**: Use Shapely for accurate spatial computations
- **Milvus Integration**: Support for Milvus cloud and local instances with URI/token authentication
- **Performance Benchmarking**: Measure query latency, throughput, and success rates
- **Accuracy Evaluation**: Compare results against ground truth with precision/recall metrics
- **Comprehensive Reports**: Generate detailed markdown and JSON evaluation reports
- **Flexible CLI**: Easy-to-use command-line interface with configuration file support

## Installation

### Prerequisites

- Python 3.12+
- uv package manager

### Install from source

```bash
git clone https://github.com/mmga-lab/milvus-geo-bench.git
cd milvus-geo-bench
make install
```

### Set up environment

```bash
# Copy configuration templates
cp config.yaml.example config.yaml
cp .env.example .env

# Edit .env file with your Milvus credentials
export MILVUS_URI="https://your-instance.api.gcp-us-west1.zillizcloud.com"
export MILVUS_TOKEN="your-api-token"
```

## Quick Start

### Full Benchmark Workflow

Run the complete benchmark pipeline:

```bash
milvus-geo-bench full-run --config config.yaml
```

This will:
1. Generate synthetic datasets
2. Load data into Milvus
3. Execute benchmark queries
4. Evaluate results and generate reports

### Individual Commands

#### Using Configuration File (Recommended)

All commands support the `--config` option to use configuration file defaults:

```bash
# 1. Generate Dataset (uses config.yaml defaults)
milvus-geo-bench --config config.yaml generate-dataset

# 2. Load Data to Milvus (uses config.yaml + environment variables)
milvus-geo-bench --config config.yaml load-data

# 3. Run Benchmark (uses config.yaml for all settings)
milvus-geo-bench --config config.yaml run-benchmark

# 4. Evaluate Results (uses config.yaml for output paths)
milvus-geo-bench --config config.yaml evaluate
```

#### Command Line Arguments (Override Config)

You can override specific config values with command line arguments:

```bash
# Override specific parameters while using config defaults
milvus-geo-bench --config config.yaml generate-dataset --num-points 50000
milvus-geo-bench --config config.yaml run-benchmark --concurrency 50
milvus-geo-bench --config config.yaml evaluate --format json
```

#### Legacy Usage (Without Config File)

```bash
# 1. Generate Dataset
milvus-geo-bench generate-dataset \
  --num-points 100000 \
  --num-queries 1000 \
  --output-dir ./data

# 2. Load Data to Milvus
milvus-geo-bench load-data \
  --uri $MILVUS_URI \
  --token $MILVUS_TOKEN \
  --collection geo_bench \
  --data-file ./data/train.parquet

# 3. Run Benchmark
milvus-geo-bench run-benchmark \
  --uri $MILVUS_URI \
  --token $MILVUS_TOKEN \
  --collection geo_bench \
  --queries ./data/test.parquet \
  --output ./data/results.parquet

# 4. Evaluate Results
milvus-geo-bench evaluate \
  --results ./data/results.parquet \
  --ground-truth ./data/ground_truth.parquet \
  --output ./reports/evaluation_report.md
```

## Configuration

### Environment Variables

```bash
MILVUS_URI=https://your-instance.api.gcp-us-west1.zillizcloud.com
MILVUS_TOKEN=your-api-token
```

### Configuration File (config.yaml)

**New: All commands now support `--config` for unified configuration!**

```yaml
# Dataset generation settings
dataset:
  num_points: 1000000          # Number of training points
  num_queries: 1000            # Number of test queries  
  output_dir: ./data           # Output directory
  bbox: [-180, -90, 180, 90]   # World coordinates [min_lon, min_lat, max_lon, max_lat]
  min_points_per_query: 100    # Minimum results per query

# Milvus connection settings
milvus:
  uri: ${MILVUS_URI}           # Environment variable substitution
  token: ${MILVUS_TOKEN}       # Environment variable substitution
  collection: geo_bench        # Collection name
  batch_size: 1000            # Batch size for data insertion
  timeout: 30                 # Connection timeout

# Benchmark execution settings
benchmark:
  timeout: 30                 # Query timeout in seconds
  warmup: 10                  # Number of warmup queries
  concurrency: 100            # Parallel query execution threads

# Output settings
output:
  results: ./data/results.parquet              # Benchmark results file
  report: ./reports/evaluation_report.md       # Evaluation report file
```

**Benefits of using config files:**
- Consistent settings across all commands
- Environment variable substitution (${VAR_NAME})
- Override specific values with CLI arguments
- Version control friendly configuration

## Data Format

The tool uses Parquet format for all data storage:

### Training Data (train.parquet)
| Column | Type | Description |
|--------|------|-------------|
| id | int64 | Unique identifier |
| wkt | string | WKT Point geometry |
| vec | list[float64] | 8-dimensional vector |

### Test Queries (test.parquet)  
| Column | Type | Description |
|--------|------|-------------|
| query_id | int64 | Query identifier |
| expr | string | ST_WITHIN expression |
| polygon_wkt | string | Query polygon WKT |
| center_lon | float64 | Polygon center longitude |
| center_lat | float64 | Polygon center latitude |
| radius | float64 | Polygon radius |

### Ground Truth (ground_truth.parquet)
| Column | Type | Description |
|--------|------|-------------|
| query_id | int64 | Query identifier |
| result_ids | list[int64] | Expected result IDs |
| result_count | int64 | Number of expected results |

### Benchmark Results (results.parquet)
| Column | Type | Description |
|--------|------|-------------|
| query_id | int64 | Query identifier |
| query_time_ms | float64 | Query execution time |
| result_ids | list[int64] | Returned result IDs |
| result_count | int64 | Number of returned results |
| success | bool | Query success status |

## Evaluation Metrics

The tool provides comprehensive accuracy and performance metrics:

### Accuracy Metrics
- **Precision**: Fraction of returned results that are correct
- **Recall**: Fraction of correct results that are returned  
- **F1-Score**: Harmonic mean of precision and recall
- **Macro/Micro averages**: Different aggregation methods

### Performance Metrics
- **Query Latency**: Response time statistics (mean, median, P95, P99)
- **Throughput**: Queries per second
- **Success Rate**: Percentage of successful queries

### Report Formats
- **Markdown**: Human-readable evaluation reports
- **JSON**: Machine-readable metrics for integration

## CLI Reference

### Global Options

```bash
# Get help
milvus-geo-bench --help

# Use configuration file with verbose logging
milvus-geo-bench --verbose --config config.yaml <command>

# All commands support the --config option
milvus-geo-bench --config config.yaml <command> [options]
```

### Commands

- `generate-dataset`: Generate training and test datasets
- `load-data`: Load data into Milvus collection
- `run-benchmark`: Execute benchmark queries
- `evaluate`: Evaluate results against ground truth
- `full-run`: Execute complete benchmark workflow

Use `--help` with any command for detailed options.

## Examples

### Using Configuration File (Recommended)

```bash
# Small test run (override config values)
milvus-geo-bench --config config.yaml generate-dataset --num-points 10000 --num-queries 100

# Geographic region test (San Francisco Bay Area)
milvus-geo-bench --config config.yaml generate-dataset --bbox "-122.5,37.0,-121.5,38.0"

# High performance benchmark
milvus-geo-bench --config config.yaml run-benchmark --timeout 60 --warmup 50 --concurrency 200

# Custom output format
milvus-geo-bench --config config.yaml evaluate --format json --output ./reports/metrics.json

# Complete workflow with config
milvus-geo-bench --config config.yaml full-run
```

### Legacy Examples (Without Config File)

```bash
# Small test run
milvus-geo-bench generate-dataset --num-points 10000 --num-queries 100

# Geographic region (San Francisco)
milvus-geo-bench generate-dataset --bbox "-122.5,37.0,-121.5,38.0"

# High performance test
milvus-geo-bench run-benchmark --uri $MILVUS_URI --token $MILVUS_TOKEN --timeout 60 --warmup 50
```

## Development

### Dependencies

The tool uses the following key dependencies:
- `pymilvus>=2.4.0`: Milvus Python client (from TestPyPI)
- `shapely>=2.0.0`: Geospatial computations
- `pandas>=2.0.0`: Data processing
- `pyarrow>=14.0.0`: Parquet file support
- `click>=8.0.0`: CLI framework
- `ruff>=0.6.0`: Code formatting and linting (dev)

### Code Quality

The project uses ruff for code formatting and linting:

```bash
# Format code
make format

# Check code quality  
make check

# Install dependencies
make install

# Install with dev dependencies
make dev-install

# Clean generated files
make clean
```

### Project Structure

```
src/milvus_geo_bench/
├── __init__.py          # CLI commands
├── dataset.py           # Dataset generation
├── milvus_client.py     # Milvus client wrapper
├── benchmark.py         # Benchmark execution
├── evaluator.py         # Result evaluation
└── utils.py             # Utility functions
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Create an issue in the repository
- Check the documentation and examples