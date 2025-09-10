# Milvus Geo Benchmark CLI Documentation

Complete documentation for the Milvus Geo Benchmark CLI tool - a comprehensive testing framework for geospatial search performance in Milvus.

## Quick Navigation

### 🚀 Getting Started
- **[Quick Start Guide](quick-start.md)** - Get running in under 10 minutes
- **[Installation Guide](quick-start.md#installation)** - Step-by-step setup instructions

### 📖 Main Documentation
- **[User Guide](user-guide.md)** - Comprehensive usage documentation
- **[API Reference](api-reference.md)** - Detailed API and module documentation

### 📋 Reference Materials
- **[Command Reference](#command-reference)** - All CLI commands and options
- **[Configuration Reference](#configuration-reference)** - Configuration file examples
- **[Data Format Reference](#data-format-reference)** - File schemas and formats

## Overview

The Milvus Geo Benchmark CLI is a specialized tool for testing and evaluating geospatial search performance in Milvus databases. It provides:

- **Synthetic Data Generation**: Creates realistic geospatial datasets with configurable parameters
- **Performance Benchmarking**: Measures query latency, throughput, and success rates
- **Accuracy Evaluation**: Compares results against ground truth with comprehensive metrics
- **Comprehensive Reporting**: Generates detailed analysis in multiple formats

### Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Dataset         │    │ Data Loading │    │ Benchmarking    │    │ Evaluation      │
│ Generation      │───▶│              │───▶│                 │───▶│                 │
│                 │    │              │    │                 │    │                 │
│ • train.parquet │    │ • Collection │    │ • Query timing  │    │ • Accuracy      │
│ • test.parquet  │    │ • Indexing   │    │ • Success rates │    │ • Performance   │
│ • ground_truth  │    │ • Data load  │    │ • Result counts │    │ • Reports       │
└─────────────────┘    └──────────────┘    └─────────────────┘    └─────────────────┘
```

## Command Reference

### Core Commands

| Command | Purpose | Documentation |
|---------|---------|---------------|
| `generate-dataset` | Create synthetic geospatial data | [User Guide](user-guide.md#1-generate-dataset) |
| `load-data` | Insert data into Milvus collection | [User Guide](user-guide.md#2-load-data) |
| `run-benchmark` | Execute performance tests | [User Guide](user-guide.md#3-run-benchmark) |
| `evaluate` | Analyze results vs ground truth | [User Guide](user-guide.md#4-evaluate) |
| `full-run` | Complete workflow in one command | [User Guide](user-guide.md#5-full-workflow) |

### Global Options

| Option | Description | Default |
|--------|-------------|---------|
| `--verbose, -v` | Enable debug logging | False |
| `--config, -c` | Configuration file path | None |
| `--help` | Show command help | - |

## Configuration Reference

### Basic Configuration

```yaml
dataset:
  num_points: 100000              # Training data size
  num_queries: 1000               # Test query count
  bbox: [-180, -90, 180, 90]     # Global bounding box
  min_points_per_query: 100       # Minimum results per query
  max_radius: 1.0                 # Maximum query radius

milvus:
  uri: "${MILVUS_URI}"           # Server connection
  token: "${MILVUS_TOKEN}"       # Authentication
  collection: "geo_bench"         # Collection name
  batch_size: 1000               # Insertion batch size

benchmark:
  timeout: 30                     # Query timeout (seconds)
  warmup: 10                     # Warmup query count
```

### Environment Variables

```bash
# Required for data loading and benchmarking
export MILVUS_URI="http://localhost:19530"
export MILVUS_TOKEN="root:Milvus"

# Optional customizations
export MILVUS_COLLECTION="geo_bench"
```

## Data Format Reference

### Training Data (`train.parquet`)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | int64 | Unique identifier | 12345 |
| `wkt` | string | Point geometry (WKT) | "POINT(-122.42 37.77)" |
| `vec` | list[float] | 8D normalized vector | [0.1, -0.2, ...] |

### Test Queries (`test.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `query_id` | int64 | Query identifier |
| `expr` | string | ST_WITHIN spatial expression |
| `polygon_wkt` | string | Query polygon (WKT) |
| `center_lon` | float64 | Query center longitude |
| `center_lat` | float64 | Query center latitude |
| `radius` | float64 | Query radius (degrees) |

### Results (`results.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `query_id` | int64 | Query identifier |
| `query_time_ms` | float64 | Execution time |
| `result_ids` | list[int64] | Matching point IDs |
| `result_count` | int64 | Number of results |
| `success` | bool | Query success status |
| `error_message` | string | Error details (if any) |

## Usage Examples

### Quick Test
```bash
# Generate small test dataset
uv run milvus-geo-bench generate-dataset \
  --num-points 1000 \
  --num-queries 5 \
  --bbox "-1,-1,1,1" \
  --output-dir ./test

# Complete workflow
export MILVUS_URI="http://localhost:19530"
export MILVUS_TOKEN="root:Milvus"

uv run milvus-geo-bench full-run --output-dir ./test
```

### Production Benchmark
```bash
# Large-scale benchmark
uv run milvus-geo-bench generate-dataset \
  --num-points 1000000 \
  --num-queries 10000 \
  --bbox "-125,25,-65,50" \
  --output-dir ./production

uv run milvus-geo-bench load-data \
  --collection production_geo \
  --data-file ./production/train.parquet \
  --batch-size 5000

uv run milvus-geo-bench run-benchmark \
  --collection production_geo \
  --queries ./production/test.parquet \
  --timeout 60 \
  --warmup 50
```

### Regional Testing
```bash
# Test specific regions
regions=(
  "california:-124,32,-114,42"
  "texas:-106,25,-93,36"
  "florida:-87,24,-79,31"
)

for region in "${regions[@]}"; do
  name="${region%%:*}"
  bbox="${region##*:}"
  
  uv run milvus-geo-bench generate-dataset \
    --bbox "$bbox" \
    --output-dir "./regions/$name"
done
```

## Performance Expectations

### With RTREE Index (Recommended)
- **Query Time**: 20-100ms per query
- **Throughput**: 10-50 QPS
- **Success Rate**: 95%+
- **Accuracy**: F1-Score > 0.90

### Without Spatial Index
- **Query Time**: 100-1000ms+ per query
- **Throughput**: 1-10 QPS
- **Resource Usage**: High (full table scan)

## Troubleshooting Quick Reference

| Issue | Quick Fix | Documentation |
|-------|-----------|---------------|
| No queries generated | Use smaller bbox: `--bbox "-1,-1,1,1"` | [User Guide](user-guide.md#troubleshooting) |
| Connection failed | Check `$MILVUS_URI` and `$MILVUS_TOKEN` | [Quick Start](quick-start.md#troubleshooting) |
| Poor performance | Verify RTREE index creation | [User Guide](user-guide.md#performance-optimization) |
| Memory issues | Reduce batch size: `--batch-size 500` | [User Guide](user-guide.md#troubleshooting) |

## Support and Contribution

### Getting Help

1. **Documentation**: Start with [Quick Start Guide](quick-start.md)
2. **API Reference**: Check [API Reference](api-reference.md) for detailed function docs
3. **Troubleshooting**: See troubleshooting sections in guides
4. **Issues**: Submit issues with detailed reproduction steps

### Development

```bash
# Development setup
make dev-install

# Code quality
make format    # Format code
make check     # Check quality
make clean     # Clean generated files
```

### Testing

```bash
# Run with minimal dataset for testing
uv run milvus-geo-bench generate-dataset \
  --num-points 100 \
  --num-queries 1 \
  --bbox "-0.1,-0.1,0.1,0.1" \
  --output-dir ./minimal_test
```

## Version Information

- **Minimum Python**: 3.12+
- **Minimum Milvus**: 2.4+ (for GEOMETRY support)
- **Recommended Milvus**: 2.7+
- **PyMilvus**: 2.7.0rc29+ (for full feature support)

## License and Attribution

This tool is designed for comprehensive geospatial search benchmarking with Milvus. It provides industry-standard metrics and evaluation methods for spatial database performance analysis.

---

**Next Steps:**
- New users: Start with [Quick Start Guide](quick-start.md)
- Comprehensive usage: Read [User Guide](user-guide.md)
- Advanced integration: Check [API Reference](api-reference.md)