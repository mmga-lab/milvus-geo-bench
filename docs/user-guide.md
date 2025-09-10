# Milvus Geo Benchmark CLI User Guide

A comprehensive guide for using the Milvus Geo Benchmark CLI tool to test and evaluate geospatial search performance.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Command Reference](#command-reference)
5. [Workflows](#workflows)
6. [Data Formats](#data-formats)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Usage](#advanced-usage)

## Overview

The Milvus Geo Benchmark CLI is a specialized tool designed to comprehensively test geospatial search capabilities in Milvus. It provides a complete pipeline from synthetic data generation to performance evaluation.

### Key Features

- **Synthetic Data Generation**: Creates realistic geospatial datasets with configurable parameters
- **Intelligent Query Generation**: Generates spatial queries with guaranteed minimum result counts
- **Performance Benchmarking**: Measures query latency, throughput, and success rates
- **Accuracy Evaluation**: Compares results against ground truth with precision/recall metrics
- **Comprehensive Reporting**: Generates detailed markdown and JSON reports
- **Modular Design**: Run individual steps or complete workflows

### Architecture

```
Dataset Generation → Data Loading → Benchmarking → Evaluation → Reporting
      ↓                  ↓             ↓            ↓           ↓
  train.parquet    Milvus Collection  results.parquet  metrics  reports/
  test.parquet     (with indexes)                               
  ground_truth.parquet                                          
```

## Installation

### Prerequisites

- **Python**: 3.12 or higher
- **uv**: Modern Python package manager ([installation guide](https://docs.astral.sh/uv/))
- **Milvus Server**: 2.7+ recommended (for GEOMETRY field support)
- **System Requirements**: 
  - 4GB+ RAM (for large datasets)
  - 2GB+ disk space (for data and results)

### Setup Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd milvus_geo_bench

# 2. Install dependencies
make install

# 3. Verify installation
uv run milvus-geo-bench --help

# 4. For development (optional)
make dev-install
```

### Verify Installation

```bash
# Check CLI is working
uv run milvus-geo-bench --version

# List all available commands
uv run milvus-geo-bench --help
```

## Configuration

### Environment Variables

The tool requires Milvus connection information. Set these environment variables:

```bash
# Required for data loading and benchmarking
export MILVUS_URI="http://your-milvus-server:19530"
export MILVUS_TOKEN="root:Milvus"  # Default Milvus credentials

# Optional: Custom collection name
export MILVUS_COLLECTION="geo_bench"
```

For different environments:

```bash
# Development
export MILVUS_URI="http://localhost:19530"
export MILVUS_TOKEN="root:Milvus"

# Production
export MILVUS_URI="https://production-milvus.company.com:443"
export MILVUS_TOKEN="username:password"

# Cloud (Zilliz Cloud)
export MILVUS_URI="https://in03-xxxxx.api.gcp-us-west1.zillizcloud.com"
export MILVUS_TOKEN="your-api-key"
```

### Configuration Files (Optional)

For complex configurations, use configuration files:

```bash
# Copy templates
cp config.yaml.example config.yaml
cp .env.example .env
```

**config.yaml example:**
```yaml
dataset:
  num_points: 100000
  num_queries: 1000
  bbox: [-180, -90, 180, 90]
  min_points_per_query: 100
  max_radius: 1.0

milvus:
  uri: "${MILVUS_URI}"
  token: "${MILVUS_TOKEN}"
  collection: "geo_bench"
  batch_size: 1000
  timeout: 30

benchmark:
  timeout: 30
  warmup: 10
```

## Command Reference

### Global Options

All commands support these global options:

- `--verbose, -v`: Enable verbose logging (DEBUG level)
- `--config, -c PATH`: Use custom configuration file
- `--help`: Show command help

### 1. generate-dataset

Generate synthetic geospatial training and test data.

**Syntax:**
```bash
milvus-geo-bench generate-dataset [OPTIONS]
```

**Options:**
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--num-points` | INTEGER | 100,000 | Number of training points to generate |
| `--num-queries` | INTEGER | 1,000 | Number of test queries to create |
| `--output-dir` | PATH | ./data | Directory for output files |
| `--bbox` | TEXT | -180,-90,180,90 | Bounding box (min_lon,min_lat,max_lon,max_lat) |
| `--min-points-per-query` | INTEGER | 100 | Minimum points each query must match |
| `--max-radius` | FLOAT | 1.0 | Maximum radius for query polygons |

**Examples:**

```bash
# Small test dataset for development
milvus-geo-bench generate-dataset \
  --num-points 1000 \
  --num-queries 10 \
  --bbox "-10,-10,10,10" \
  --min-points-per-query 50 \
  --output-dir ./test_data

# Production dataset with specific region (California)
milvus-geo-bench generate-dataset \
  --num-points 500000 \
  --num-queries 5000 \
  --bbox "-124.7,32.5,-114.1,42.0" \
  --min-points-per-query 200 \
  --output-dir ./california_data

# Quick benchmark dataset
milvus-geo-bench generate-dataset \
  --num-points 10000 \
  --num-queries 100 \
  --bbox "-1,-1,1,1" \
  --output-dir ./quick_test
```

**Generated Files:**
- `train.parquet`: Training data (points + vectors)
- `test.parquet`: Test queries (spatial expressions)
- `ground_truth.parquet`: Expected results

**Performance Notes:**
- Larger bounding boxes may result in fewer successful queries
- Smaller `min-points-per-query` increases query generation success
- Generation time scales linearly with `num-points`

### 2. load-data

Insert training data into a Milvus collection with proper schema and indexing.

**Syntax:**
```bash
milvus-geo-bench load-data [OPTIONS]
```

**Options:**
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--uri` | TEXT | $MILVUS_URI | Milvus server URI |
| `--token` | TEXT | $MILVUS_TOKEN | Authentication token |
| `--collection` | TEXT | geo_bench | Collection name |
| `--data-file` | PATH | required | Training data parquet file |
| `--batch-size` | INTEGER | 1,000 | Batch size for insertion |
| `--recreate/--no-recreate` | FLAG | True | Recreate collection if exists |

**Examples:**

```bash
# Basic data loading
milvus-geo-bench load-data \
  --uri "http://localhost:19530" \
  --token "root:Milvus" \
  --collection geo_test \
  --data-file ./data/train.parquet

# High-performance loading with large batches
milvus-geo-bench load-data \
  --data-file ./big_dataset/train.parquet \
  --collection production_geo \
  --batch-size 5000 \
  --recreate

# Loading without recreating collection
milvus-geo-bench load-data \
  --data-file ./additional_data/train.parquet \
  --collection existing_collection \
  --no-recreate
```

**What Happens:**
1. **Connection**: Connects to Milvus server
2. **Schema Creation**: Creates collection with fields:
   - `id` (INT64, primary key)
   - `location` (GEOMETRY, for spatial data)
   - `embedding` (FLOAT_VECTOR, dim=8)
3. **Index Creation**: 
   - RTREE index on `location` field
   - IVF_FLAT index on `embedding` field
4. **Data Insertion**: Batch insertion with progress tracking
5. **Collection Loading**: Loads collection to memory

**Performance Tips:**
- Use batch sizes 1000-5000 for optimal performance
- Larger batches = faster insertion but more memory usage
- Monitor disk space during insertion

### 3. run-benchmark

Execute performance tests on loaded collections.

**Syntax:**
```bash
milvus-geo-bench run-benchmark [OPTIONS]
```

**Options:**
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--uri` | TEXT | $MILVUS_URI | Milvus server URI |
| `--token` | TEXT | $MILVUS_TOKEN | Authentication token |
| `--collection` | TEXT | geo_bench | Collection to query |
| `--queries` | PATH | required | Test queries parquet file |
| `--output` | PATH | ./data/results.parquet | Results output file |
| `--timeout` | INTEGER | 30 | Query timeout (seconds) |
| `--warmup` | INTEGER | 10 | Number of warmup queries |

**Examples:**

```bash
# Standard benchmark
milvus-geo-bench run-benchmark \
  --uri "http://localhost:19530" \
  --token "root:Milvus" \
  --collection geo_test \
  --queries ./data/test.parquet \
  --output ./data/results.parquet

# High-performance benchmark with extended timeout
milvus-geo-bench run-benchmark \
  --collection production_geo \
  --queries ./large_dataset/test.parquet \
  --output ./results/prod_benchmark.parquet \
  --timeout 60 \
  --warmup 20

# Quick benchmark for development
milvus-geo-bench run-benchmark \
  --collection quick_test \
  --queries ./test_data/test.parquet \
  --timeout 10 \
  --warmup 3
```

**Benchmark Process:**
1. **Health Check**: Verifies Milvus connection and collection
2. **Query Loading**: Loads test queries from parquet file
3. **Warmup Phase**: Runs warmup queries (not measured)
4. **Benchmark Execution**: Times each query with detailed metrics
5. **Results Saving**: Stores results with timing and accuracy data

**Measured Metrics:**
- Query execution time (milliseconds)
- Result count per query
- Success/failure status
- Error messages (if any)
- Throughput (queries per second)

### 4. evaluate

Compare benchmark results against ground truth for accuracy analysis.

**Syntax:**
```bash
milvus-geo-bench evaluate [OPTIONS]
```

**Options:**
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--results` | PATH | required | Benchmark results parquet file |
| `--ground-truth` | PATH | required | Ground truth parquet file |
| `--output` | PATH | ./reports/evaluation_report.md | Report output path |
| `--format` | CHOICE | markdown | Output format (markdown/json) |
| `--print-summary/--no-print-summary` | FLAG | True | Print console summary |

**Examples:**

```bash
# Standard evaluation with markdown report
milvus-geo-bench evaluate \
  --results ./data/results.parquet \
  --ground-truth ./data/ground_truth.parquet \
  --output ./reports/accuracy_report.md

# JSON output for programmatic processing
milvus-geo-bench evaluate \
  --results ./benchmark/results.parquet \
  --ground-truth ./benchmark/ground_truth.parquet \
  --output ./api/metrics.json \
  --format json

# Evaluation without console output
milvus-geo-bench evaluate \
  --results ./data/results.parquet \
  --ground-truth ./data/ground_truth.parquet \
  --output ./silent_report.md \
  --no-print-summary
```

**Evaluation Metrics:**

**Accuracy Metrics:**
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)  
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)
- **Macro/Micro Averages**: Different averaging methods

**Performance Metrics:**
- Query time statistics (mean, median, percentiles)
- Result count distributions
- Success rates
- Throughput measurements

**Statistical Analysis:**
- Distribution analysis (std dev, quartiles)
- Confusion matrix summary
- Detailed per-query breakdown

### 5. full-run

Execute the complete benchmark workflow in a single command.

**Syntax:**
```bash
milvus-geo-bench full-run [OPTIONS]
```

**Options:**
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | PATH | None | Configuration file override |
| `--output-dir` | PATH | ./data | Directory for datasets |
| `--reports-dir` | PATH | ./reports | Directory for reports |

**Examples:**

```bash
# Complete workflow with defaults
export MILVUS_URI="http://localhost:19530"
export MILVUS_TOKEN="root:Milvus"
milvus-geo-bench full-run

# Custom directories
milvus-geo-bench full-run \
  --output-dir ./benchmark_2024 \
  --reports-dir ./benchmark_2024/reports

# With custom configuration
milvus-geo-bench full-run \
  --config ./configs/production.yaml \
  --output-dir ./prod_benchmark \
  --reports-dir ./prod_reports
```

**Workflow Steps:**
1. **Dataset Generation** (using config defaults)
2. **Data Loading** (creates collection and indexes)
3. **Benchmark Execution** (runs performance tests)
4. **Result Evaluation** (generates accuracy reports)
5. **Cleanup** (closes connections)

**Generated Outputs:**
- All dataset files in `output-dir`
- Benchmark results in `output-dir`
- Evaluation reports in `reports-dir` (markdown + JSON)

## Workflows

### Development Workflow

For development and testing with small datasets:

```bash
# 1. Generate small test dataset
milvus-geo-bench generate-dataset \
  --num-points 1000 \
  --num-queries 10 \
  --bbox "-1,-1,1,1" \
  --min-points-per-query 30 \
  --output-dir ./dev_test

# 2. Load data
milvus-geo-bench load-data \
  --collection dev_test \
  --data-file ./dev_test/train.parquet

# 3. Run quick benchmark
milvus-geo-bench run-benchmark \
  --collection dev_test \
  --queries ./dev_test/test.parquet \
  --output ./dev_test/results.parquet \
  --timeout 10 \
  --warmup 2

# 4. Evaluate results
milvus-geo-bench evaluate \
  --results ./dev_test/results.parquet \
  --ground-truth ./dev_test/ground_truth.parquet \
  --output ./dev_test/report.md
```

### Production Benchmark Workflow

For comprehensive production testing:

```bash
# Set production environment
export MILVUS_URI="https://prod-milvus.company.com:443"
export MILVUS_TOKEN="prod-token"

# 1. Generate large dataset
milvus-geo-bench generate-dataset \
  --num-points 1000000 \
  --num-queries 10000 \
  --bbox "-125,25,-65,50" \
  --min-points-per-query 500 \
  --output-dir ./prod_benchmark

# 2. Load with high performance settings
milvus-geo-bench load-data \
  --collection production_geo_bench \
  --data-file ./prod_benchmark/train.parquet \
  --batch-size 5000

# 3. Run comprehensive benchmark
milvus-geo-bench run-benchmark \
  --collection production_geo_bench \
  --queries ./prod_benchmark/test.parquet \
  --output ./prod_benchmark/results.parquet \
  --timeout 60 \
  --warmup 50

# 4. Generate detailed reports
milvus-geo-bench evaluate \
  --results ./prod_benchmark/results.parquet \
  --ground-truth ./prod_benchmark/ground_truth.parquet \
  --output ./prod_benchmark/detailed_report.md

# 5. Also generate JSON for automated processing
milvus-geo-bench evaluate \
  --results ./prod_benchmark/results.parquet \
  --ground-truth ./prod_benchmark/ground_truth.parquet \
  --output ./prod_benchmark/metrics.json \
  --format json
```

### Comparative Analysis Workflow

Comparing different Milvus configurations or versions:

```bash
# Generate shared dataset
milvus-geo-bench generate-dataset \
  --num-points 100000 \
  --num-queries 1000 \
  --output-dir ./comparison_data

# Test Configuration A
milvus-geo-bench load-data \
  --collection config_a \
  --data-file ./comparison_data/train.parquet

milvus-geo-bench run-benchmark \
  --collection config_a \
  --queries ./comparison_data/test.parquet \
  --output ./comparison_data/results_a.parquet

# Test Configuration B  
milvus-geo-bench load-data \
  --collection config_b \
  --data-file ./comparison_data/train.parquet

milvus-geo-bench run-benchmark \
  --collection config_b \
  --queries ./comparison_data/test.parquet \
  --output ./comparison_data/results_b.parquet

# Evaluate both
milvus-geo-bench evaluate \
  --results ./comparison_data/results_a.parquet \
  --ground-truth ./comparison_data/ground_truth.parquet \
  --output ./reports/config_a_report.md

milvus-geo-bench evaluate \
  --results ./comparison_data/results_b.parquet \
  --ground-truth ./comparison_data/ground_truth.parquet \
  --output ./reports/config_b_report.md
```

## Data Formats

### Training Data Format

**File:** `train.parquet`

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | int64 | Unique identifier | 1, 2, 3, ... |
| `wkt` | string | Point geometry in WKT | "POINT(-122.4194 37.7749)" |
| `vec` | list[float] | 8D normalized vector | [0.1, -0.2, 0.3, ...] |

**Sample Data:**
```python
{
    "id": 12345,
    "wkt": "POINT(-73.9857 40.7484)",  # New York City
    "vec": [0.12, -0.34, 0.56, -0.78, 0.23, -0.45, 0.67, -0.89]
}
```

### Test Queries Format

**File:** `test.parquet`

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `query_id` | int64 | Query identifier | 1, 2, 3, ... |
| `expr` | string | Milvus spatial expression | "ST_WITHIN(location, 'POLYGON(...)')" |
| `polygon_wkt` | string | Query polygon in WKT | "POLYGON((-1 -1, 1 -1, ...))" |
| `center_lon` | float64 | Query center longitude | -122.4194 |
| `center_lat` | float64 | Query center latitude | 37.7749 |
| `radius` | float64 | Query radius (degrees) | 0.01 |

**Sample Query:**
```python
{
    "query_id": 1,
    "expr": "ST_WITHIN(location, 'POLYGON((-122.42 37.77, -122.41 37.78, ...))')",
    "polygon_wkt": "POLYGON((-122.42 37.77, -122.41 37.78, ...))",
    "center_lon": -122.4194,
    "center_lat": 37.7749,
    "radius": 0.005
}
```

### Ground Truth Format

**File:** `ground_truth.parquet`

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `query_id` | int64 | Query identifier | 1, 2, 3, ... |
| `result_ids` | list[int64] | Expected matching point IDs | [123, 456, 789] |
| `result_count` | int64 | Number of expected results | 3 |

### Benchmark Results Format

**File:** `results.parquet`

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `query_id` | int64 | Query identifier | 1, 2, 3, ... |
| `query_time_ms` | float64 | Execution time in milliseconds | 25.347 |
| `result_ids` | list[int64] | Actual matching point IDs | [123, 456] |
| `result_count` | int64 | Number of actual results | 2 |
| `success` | bool | Query execution success | true |
| `error_message` | string | Error details if failed | null |

## Performance Optimization

### Index Configuration

The tool creates optimized indexes for geospatial queries:

**RTREE Index (Spatial):**
```python
{
    "field_name": "location",
    "index_type": "RTREE",
    "params": {}
}
```

**Vector Index:**
```python
{
    "field_name": "embedding", 
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}
```

### Performance Tuning Parameters

**Data Loading:**
- `batch_size`: 1000-5000 for optimal insertion speed
- Higher batch sizes use more memory but are faster
- Monitor system resources during large insertions

**Query Execution:**
- `warmup`: 10-50 queries to stabilize cache
- `timeout`: 30-60 seconds depending on data size
- Use connection pooling for concurrent access

**Dataset Generation:**
- Smaller bounding boxes improve query success rates
- `min_points_per_query`: Balance between query success and difficulty
- `max_radius`: Prevent overly large query regions

### Expected Performance Metrics

**With RTREE Index:**
- Query time: 20-100ms (depending on result set size)
- Throughput: 10-50 QPS (queries per second)
- Success rate: 95%+ for well-configured datasets

**Without Spatial Index:**
- Query time: 100-1000ms+ (full scan)
- Much lower throughput
- Higher resource usage

### Memory and Storage Requirements

**Memory Usage:**
- Collection loading: ~2-3x data size in RAM
- Query processing: Additional 10-20% for indexes
- Batch insertion: Batch size × record size

**Storage Requirements:**
- Raw data: ~1KB per point (with vector)
- Indexes: ~30-50% of data size
- Results and reports: ~10% of data size

## Troubleshooting

### Common Issues and Solutions

#### 1. Dataset Generation Issues

**Problem:** No queries generated
```
WARNING - Only generated 0 queries out of 1000 requested
```

**Solutions:**
```bash
# Reduce minimum points requirement
--min-points-per-query 50

# Use smaller, denser bounding box
--bbox "-1,-1,1,1"

# Increase training points
--num-points 50000

# Increase maximum radius
--max-radius 5.0
```

**Problem:** Query generation very slow

**Solutions:**
```bash
# Reduce spatial complexity
--num-queries 100

# Use focused geographic region
--bbox "-10,-10,10,10"
```

#### 2. Connection Issues

**Problem:** Connection failed
```
ERROR - Failed to connect to Milvus: Connection refused
```

**Solutions:**
```bash
# Check Milvus server status
curl http://your-milvus:19530/health

# Verify environment variables
echo $MILVUS_URI
echo $MILVUS_TOKEN

# Test basic connectivity
telnet your-milvus-host 19530
```

**Problem:** Authentication failed
```
ERROR - Authentication failed
```

**Solutions:**
```bash
# Verify credentials
export MILVUS_TOKEN="correct-username:correct-password"

# For Zilliz Cloud, use API key
export MILVUS_TOKEN="your-api-key"

# Check token format (no spaces, correct separator)
```

#### 3. Data Loading Issues

**Problem:** Collection creation failed
```
ERROR - Failed to create collection: GEOMETRY type not supported
```

**Solutions:**
```bash
# Upgrade Milvus to 2.4+
# Or upgrade pymilvus
uv add --index-url https://test.pypi.org/simple/ pymilvus==2.7.0rc29

# Check Milvus version
curl http://your-milvus:19530/health
```

**Problem:** Index creation failed
```
ERROR - Failed to create indexes: invalid index type: RTREE
```

**Solutions:**
```bash
# Verify Milvus version supports RTREE
# Check server logs for detailed error
# Try without geometry index (reduced performance)
```

#### 4. Benchmark Issues

**Problem:** All queries timeout
```
ERROR - Query timeout after 30 seconds
```

**Solutions:**
```bash
# Increase timeout
--timeout 60

# Check system resources (CPU, memory)
# Verify collection is loaded
# Reduce query complexity

# Check if indexes are built
```

**Problem:** Poor performance
```
INFO - Average query time: 500.00ms
```

**Solutions:**
```bash
# Verify RTREE index was created
# Check collection loading status
# Monitor system resources
# Reduce result set sizes
```

#### 5. Evaluation Issues

**Problem:** No matching queries
```
ERROR - No matching queries found between results and ground truth
```

**Solutions:**
```bash
# Check query_id alignment
# Verify file formats
# Ensure successful benchmark results exist

# Debug with smaller dataset first
```

**Problem:** Column not found errors
```
ERROR - KeyError: 'result_ids'
```

**Solutions:**
```bash
# Regenerate results with current tool version
# Check parquet file schema
# Verify benchmark completed successfully
```

### Performance Troubleshooting

#### Slow Query Performance

1. **Check Index Status:**
```bash
# Connect to Milvus and verify indexes
```

2. **Monitor Resources:**
```bash
# Check CPU, memory, disk I/O
top
iostat -x 1
```

3. **Optimize Queries:**
```bash
# Reduce result set sizes
--min-points-per-query 50

# Use smaller query regions
--max-radius 0.5
```

#### Memory Issues

1. **Reduce Batch Size:**
```bash
--batch-size 1000  # Instead of 5000
```

2. **Monitor Collection Loading:**
```bash
# Check Milvus memory usage
# Ensure adequate system RAM
```

#### Disk Space Issues

1. **Clean Old Data:**
```bash
make clean
rm -rf old_benchmark_*
```

2. **Monitor Space:**
```bash
df -h
du -sh data/ reports/
```

### Debug Mode

Enable verbose logging for detailed troubleshooting:

```bash
# Enable debug logging
milvus-geo-bench --verbose generate-dataset ...
milvus-geo-bench -v load-data ...

# Check log files
tail -f /var/log/milvus/milvus.log  # Milvus server logs
```

### Getting Help

1. **Check Tool Version:**
```bash
uv run milvus-geo-bench --version
```

2. **Verify Dependencies:**
```bash
uv run python -c "import pymilvus; print(pymilvus.__version__)"
```

3. **Test with Minimal Example:**
```bash
# Start with smallest possible dataset
milvus-geo-bench generate-dataset \
  --num-points 100 \
  --num-queries 1 \
  --bbox "-0.1,-0.1,0.1,0.1" \
  --output-dir ./minimal_test
```

## Advanced Usage

### Custom Configuration Files

Create specialized configurations for different use cases:

**development.yaml:**
```yaml
dataset:
  num_points: 1000
  num_queries: 10
  bbox: [-1, -1, 1, 1]
  min_points_per_query: 10
  max_radius: 0.5

milvus:
  collection: "dev_test"
  batch_size: 100
  timeout: 10

benchmark:
  timeout: 10
  warmup: 2
```

**production.yaml:**
```yaml
dataset:
  num_points: 1000000
  num_queries: 10000
  bbox: [-125, 25, -65, 50]  # Continental US
  min_points_per_query: 500
  max_radius: 2.0

milvus:
  collection: "production_geo_bench"
  batch_size: 5000
  timeout: 60

benchmark:
  timeout: 60
  warmup: 50
```

Usage:
```bash
milvus-geo-bench full-run --config ./configs/production.yaml
```

### Batch Processing Multiple Regions

Test different geographic regions systematically:

```bash
# Define regions
regions=(
  "california:-124,32,-114,42"
  "texas:-106,25,-93,36"  
  "florida:-87,24,-79,31"
  "newyork:-80,40,-71,45"
)

# Process each region
for region in "${regions[@]}"; do
  name="${region%%:*}"
  bbox="${region##*:}"
  
  echo "Processing region: $name"
  
  milvus-geo-bench generate-dataset \
    --num-points 50000 \
    --num-queries 500 \
    --bbox "$bbox" \
    --output-dir "./regions/$name"
    
  milvus-geo-bench load-data \
    --collection "geo_$name" \
    --data-file "./regions/$name/train.parquet"
    
  milvus-geo-bench run-benchmark \
    --collection "geo_$name" \
    --queries "./regions/$name/test.parquet" \
    --output "./regions/$name/results.parquet"
    
  milvus-geo-bench evaluate \
    --results "./regions/$name/results.parquet" \
    --ground-truth "./regions/$name/ground_truth.parquet" \
    --output "./reports/$name-report.md"
done
```

### Automated Performance Regression Testing

Set up automated testing for CI/CD:

```bash
#!/bin/bash
# performance_regression_test.sh

set -e

# Configuration
TEST_NAME="regression_$(date +%Y%m%d_%H%M%S)"
BASELINE_TIME=50.0  # milliseconds
BASELINE_ACCURACY=0.85  # F1 score

# Run benchmark
milvus-geo-bench full-run \
  --output-dir "./regression/$TEST_NAME" \
  --reports-dir "./regression/$TEST_NAME/reports"

# Parse results (requires jq)
RESULTS_FILE="./regression/$TEST_NAME/reports/evaluation_metrics.json"
MEAN_TIME=$(jq '.performance.query_time.mean' "$RESULTS_FILE")
F1_SCORE=$(jq '.accuracy.macro_f1' "$RESULTS_FILE")

# Check regression
if (( $(echo "$MEAN_TIME > $BASELINE_TIME * 1.1" | bc -l) )); then
  echo "PERFORMANCE REGRESSION: Mean time $MEAN_TIME > threshold $(echo "$BASELINE_TIME * 1.1" | bc -l)"
  exit 1
fi

if (( $(echo "$F1_SCORE < $BASELINE_ACCURACY" | bc -l) )); then
  echo "ACCURACY REGRESSION: F1 score $F1_SCORE < threshold $BASELINE_ACCURACY" 
  exit 1
fi

echo "Performance test passed: time=$MEAN_TIME, accuracy=$F1_SCORE"
```

### Integration with Monitoring Systems

Export metrics to external monitoring:

```bash
# Generate JSON metrics
milvus-geo-bench evaluate \
  --results ./data/results.parquet \
  --ground-truth ./data/ground_truth.parquet \
  --output ./metrics.json \
  --format json

# Send to monitoring system (example)
curl -X POST "http://monitoring.company.com/metrics" \
  -H "Content-Type: application/json" \
  -d @./metrics.json
```

### Custom Index Configurations

For advanced users who need different index types:

```python
# Modify milvus_client.py for custom indexes
# Example: Using HNSW instead of IVF_FLAT

def _create_indexes(self, collection_name: str) -> None:
    index_params = IndexParams()
    
    # Custom vector index
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="L2", 
        params={"M": 16, "efConstruction": 200}
    )
    
    # Keep RTREE for geometry
    index_params.add_index(
        field_name="location",
        index_type="RTREE"
    )
```

This comprehensive guide covers all aspects of using the Milvus Geo Benchmark CLI tool. For additional support or feature requests, consult the project documentation or submit issues through the appropriate channels.