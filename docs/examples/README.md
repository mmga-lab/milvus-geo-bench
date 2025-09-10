# Configuration Examples

This directory contains example configuration files for different use cases.

## Available Configurations

### [quick-test.yaml](quick-test.yaml)
**Use case**: Rapid testing, CI/CD, debugging
- **Dataset**: 500 points, 5 queries
- **Region**: 1°×1° test area
- **Time**: ~30 seconds total
- **Purpose**: Quick validation, automated testing

```bash
uv run milvus-geo-bench full-run --config ./docs/examples/quick-test.yaml
```

### [development.yaml](development.yaml)
**Use case**: Development, feature testing
- **Dataset**: 1,000 points, 10 queries  
- **Region**: 2°×2° development area
- **Time**: ~2 minutes total
- **Purpose**: Local development, debugging

```bash
uv run milvus-geo-bench full-run --config ./docs/examples/development.yaml
```

### [regional.yaml](regional.yaml)
**Use case**: Regional performance testing
- **Dataset**: 100,000 points, 1,000 queries
- **Region**: Configurable (California by default)
- **Time**: ~15 minutes total
- **Purpose**: Geographic-specific testing

```bash
uv run milvus-geo-bench full-run --config ./docs/examples/regional.yaml
```

### [production.yaml](production.yaml)
**Use case**: Comprehensive production benchmarking
- **Dataset**: 1,000,000 points, 10,000 queries
- **Region**: Continental US
- **Time**: ~60+ minutes total
- **Purpose**: Full-scale performance evaluation

```bash
uv run milvus-geo-bench full-run --config ./docs/examples/production.yaml
```

## Usage Examples

### Quick Validation
```bash
# Test basic functionality quickly
export MILVUS_URI="http://localhost:19530"
export MILVUS_TOKEN="root:Milvus"

uv run milvus-geo-bench full-run --config ./docs/examples/quick-test.yaml
```

### Development Testing
```bash
# Development testing with moderate dataset
uv run milvus-geo-bench full-run \
  --config ./docs/examples/development.yaml \
  --output-dir ./dev_results \
  --reports-dir ./dev_reports
```

### Regional Comparison
```bash
# Test multiple regions
regions=("california" "texas" "florida")
base_config="./docs/examples/regional.yaml"

for region in "${regions[@]}"; do
    # Customize config per region and run
    sed "s/california/$region/g" "$base_config" > "./temp_${region}.yaml"
    
    uv run milvus-geo-bench full-run \
      --config "./temp_${region}.yaml" \
      --output-dir "./regions/$region" \
      --reports-dir "./reports/$region"
      
    rm "./temp_${region}.yaml"
done
```

### Performance Regression Testing
```bash
#!/bin/bash
# regression_test.sh

# Run standardized test
uv run milvus-geo-bench full-run \
  --config ./docs/examples/development.yaml \
  --output-dir "./regression/$(date +%Y%m%d)" \
  --reports-dir "./regression_reports/$(date +%Y%m%d)"

# Parse results and check thresholds
# (Add your specific threshold checking logic here)
```

## Customization Guide

### Modifying Bounding Boxes

Common geographic regions:

```yaml
# Major US Cities
bbox: [-74.3, 40.4, -73.7, 40.9]    # New York City
bbox: [-118.7, 33.8, -118.1, 34.4]  # Los Angeles  
bbox: [-87.9, 41.6, -87.5, 42.0]    # Chicago
bbox: [-122.7, 37.6, -122.3, 37.9]  # San Francisco

# US States
bbox: [-124, 32, -114, 42]           # California
bbox: [-106, 25, -93, 36]            # Texas
bbox: [-87, 24, -79, 31]             # Florida

# International
bbox: [-0.5, 51.3, 0.2, 51.7]       # London, UK
bbox: [2.1, 48.7, 2.5, 49.0]        # Paris, France
bbox: [116.1, 39.7, 117.0, 40.2]    # Beijing, China
```

### Performance Tuning Parameters

**For faster testing:**
```yaml
dataset:
  num_points: 1000      # Reduce points
  num_queries: 10       # Reduce queries
  min_points_per_query: 20  # Lower threshold

benchmark:
  warmup: 1            # Minimal warmup
  timeout: 10          # Short timeout
```

**For comprehensive testing:**
```yaml
dataset:
  num_points: 1000000  # More points
  num_queries: 10000   # More queries  
  min_points_per_query: 500  # Higher threshold

benchmark:
  warmup: 50           # Extensive warmup
  timeout: 120         # Extended timeout
```

### Environment Variable Usage

All configs support environment variable substitution:

```yaml
milvus:
  uri: "${MILVUS_URI}"                        # Required
  token: "${MILVUS_TOKEN}"                    # Required
  collection: "${COLLECTION_NAME:geo_bench}" # Optional with default
  batch_size: "${BATCH_SIZE:1000}"           # Optional with default
```

Set environment variables:
```bash
export MILVUS_URI="http://your-server:19530"
export MILVUS_TOKEN="your-token"
export COLLECTION_NAME="custom_collection"
export BATCH_SIZE="2000"
```

### Creating Custom Configurations

1. **Copy a base configuration:**
   ```bash
   cp ./docs/examples/development.yaml ./my-config.yaml
   ```

2. **Modify parameters:**
   ```yaml
   # Edit my-config.yaml with your specific requirements
   dataset:
     bbox: [your, custom, bounding, box]
     num_points: your_point_count
   ```

3. **Test configuration:**
   ```bash
   uv run milvus-geo-bench full-run --config ./my-config.yaml
   ```

4. **Validate results:**
   ```bash
   # Check generated files and reports
   ls -la ./your_output_dir/
   cat ./your_reports_dir/evaluation_report.md
   ```

## Configuration Validation

Test your configuration before running full benchmarks:

```bash
# Validate configuration syntax
python -c "import yaml; yaml.safe_load(open('your-config.yaml'))"

# Test with minimal dataset first
# (Temporarily reduce num_points and num_queries)

# Check environment variables
echo $MILVUS_URI
echo $MILVUS_TOKEN
```

## Best Practices

1. **Start Small**: Begin with quick-test.yaml and scale up
2. **Environment Variables**: Use environment variables for credentials
3. **Regional Focus**: Use smaller bounding boxes for better results
4. **Incremental Testing**: Test each component before full workflow
5. **Resource Monitoring**: Monitor system resources during large benchmarks
6. **Backup Results**: Save results and reports for comparison

---

These configurations cover common use cases from quick validation to comprehensive production benchmarking. Choose the appropriate configuration based on your testing needs and available resources.