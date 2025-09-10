# Quick Start Guide

Get up and running with the Milvus Geo Benchmark CLI tool in under 10 minutes.

## Prerequisites

- Python 3.12+
- [uv package manager](https://docs.astral.sh/uv/)
- Access to a Milvus server (local or remote)

## 1. Installation

```bash
# Clone and install
git clone <repository-url>
cd milvus_geo_bench
make install

# Verify installation
uv run milvus-geo-bench --help
```

## 2. Setup Milvus Connection

Set your Milvus connection details:

```bash
# For local Milvus
export MILVUS_URI="http://localhost:19530"
export MILVUS_TOKEN="root:Milvus"

# For remote Milvus
export MILVUS_URI="http://your-milvus-server:19530"
export MILVUS_TOKEN="username:password"

# For Zilliz Cloud
export MILVUS_URI="https://in03-xxx.api.gcp-us-west1.zillizcloud.com"
export MILVUS_TOKEN="your-api-key"
```

## 3. Quick Test (5 minutes)

Run a quick test with a small dataset:

```bash
# Generate small test dataset (1000 points, 5 queries)
uv run milvus-geo-bench generate-dataset \
  --num-points 1000 \
  --num-queries 5 \
  --bbox "-1,-1,1,1" \
  --min-points-per-query 30 \
  --output-dir ./quick_test

# Load data into Milvus
uv run milvus-geo-bench load-data \
  --collection quick_test \
  --data-file ./quick_test/train.parquet

# Run benchmark
uv run milvus-geo-bench run-benchmark \
  --collection quick_test \
  --queries ./quick_test/test.parquet \
  --output ./quick_test/results.parquet \
  --timeout 10 \
  --warmup 2

# Evaluate results
uv run milvus-geo-bench evaluate \
  --results ./quick_test/results.parquet \
  --ground-truth ./quick_test/ground_truth.parquet \
  --output ./quick_test/report.md
```

**Expected Output:**
- Dataset generation: ~2 seconds
- Data loading: ~5 seconds (1000 points)
- Benchmark: ~3 seconds (5 queries)
- Evaluation: ~1 second

**Check Results:**
```bash
# View performance summary
cat ./quick_test/report.md | head -20

# Check generated files
ls -la ./quick_test/
```

## 4. Full Workflow (One Command)

For a complete benchmark using default parameters:

```bash
# Set environment variables
export MILVUS_URI="http://localhost:19530"
export MILVUS_TOKEN="root:Milvus"

# Run complete workflow (this will take longer - uses 100K points)
uv run milvus-geo-bench full-run \
  --output-dir ./full_benchmark \
  --reports-dir ./full_reports
```

**Note:** The full workflow uses 100,000 points and 1,000 queries by default. This may take 10-30 minutes depending on your system.

## 5. Understanding the Results

### Performance Metrics

Your evaluation report will show:

```markdown
=== Benchmark Evaluation Summary ===
Total Queries: 5
Successful Queries: 5
Success Rate: 100.00%

Accuracy Metrics:
  Macro Precision: 1.0000
  Macro Recall: 0.8500
  Macro F1-Score: 0.9200

Performance:
  Average Query Time: 25.30ms
  Median Query Time: 22.15ms
  P95 Query Time: 35.20ms
  Throughput: 40.25 QPS
```

**What This Means:**
- **Success Rate**: Percentage of queries that completed successfully
- **Precision**: Accuracy of returned results (1.0 = perfect, no false positives)
- **Recall**: Completeness of results (0.85 = found 85% of expected results)
- **F1-Score**: Balanced accuracy metric (harmonic mean of precision and recall)
- **Query Time**: Time to execute each spatial query
- **Throughput**: Queries processed per second (QPS)

### Good Performance Indicators

✅ **Excellent Performance:**
- Success rate: 95%+
- F1-Score: 0.90+
- Query time: < 50ms
- Throughput: > 20 QPS

✅ **Good Performance:**
- Success rate: 90%+
- F1-Score: 0.80+
- Query time: < 100ms
- Throughput: > 10 QPS

⚠️ **Needs Investigation:**
- Success rate: < 90%
- F1-Score: < 0.80
- Query time: > 200ms
- Throughput: < 5 QPS

## 6. Common Next Steps

### Test Different Regions

```bash
# Test with specific geographic regions
regions=(
  "california:-124,32,-114,42"
  "newyork:-80,40,-71,45"
  "london:-0.5,51.3,0.2,51.7"
)

for region in "${regions[@]}"; do
  name="${region%%:*}"
  bbox="${region##*:}"
  
  uv run milvus-geo-bench generate-dataset \
    --num-points 10000 \
    --num-queries 50 \
    --bbox "$bbox" \
    --output-dir "./regions/$name"
done
```

### Compare Index Performance

```bash
# Test with different collection configurations
# Collection A: With RTREE index (default)
uv run milvus-geo-bench load-data \
  --collection geo_with_index \
  --data-file ./data/train.parquet

# Benchmark both and compare results
```

### Scale Testing

```bash
# Test with larger datasets
uv run milvus-geo-bench generate-dataset \
  --num-points 100000 \
  --num-queries 1000 \
  --output-dir ./scale_test

# Monitor performance scaling
```

## 7. Troubleshooting

### Connection Issues

```bash
# Test Milvus connection
curl http://your-milvus-server:19530/health

# Verify environment variables
echo $MILVUS_URI
echo $MILVUS_TOKEN
```

### No Queries Generated

```bash
# Use smaller bounding box and fewer minimum points
uv run milvus-geo-bench generate-dataset \
  --bbox "-0.1,-0.1,0.1,0.1" \
  --min-points-per-query 10
```

### Poor Performance

```bash
# Check if RTREE index was created successfully
# Look for this in the load-data output:
# "INFO - Created RTREE geometry index for collection_name"
```

### Memory Issues

```bash
# Reduce batch size
uv run milvus-geo-bench load-data \
  --batch-size 500 \
  --data-file ./data/train.parquet
```

## 8. What's Next?

- **[User Guide](user-guide.md)**: Comprehensive usage documentation
- **[API Reference](api-reference.md)**: Detailed API documentation
- **[Examples](examples/)**: Advanced usage examples

### Custom Configurations

Create a configuration file for repeated tests:

```yaml
# config.yaml
dataset:
  num_points: 50000
  num_queries: 500
  bbox: [-2, -2, 2, 2]
  min_points_per_query: 100

milvus:
  collection: "custom_bench"
  batch_size: 2000
  timeout: 60

benchmark:
  timeout: 60
  warmup: 20
```

```bash
# Use custom configuration
uv run milvus-geo-bench full-run --config ./config.yaml
```

### Performance Monitoring

Set up automated benchmarks:

```bash
#!/bin/bash
# daily_benchmark.sh
DATE=$(date +%Y%m%d)
uv run milvus-geo-bench full-run \
  --output-dir "./benchmarks/$DATE" \
  --reports-dir "./reports/$DATE"
```

---

**Congratulations!** You've successfully set up and run your first Milvus geo benchmark. The tool provides comprehensive testing of spatial query performance and accuracy, helping you optimize your geospatial applications.

For detailed documentation and advanced features, check the [User Guide](user-guide.md) and [API Reference](api-reference.md).