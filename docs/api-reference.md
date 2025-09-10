# Milvus Geo Benchmark API Reference

Complete API reference for the Milvus Geo Benchmark CLI tool modules and functions.

## Table of Contents

1. [CLI Commands](#cli-commands)
2. [Core Modules](#core-modules)
3. [Data Types](#data-types)
4. [Configuration Schema](#configuration-schema)
5. [Error Handling](#error-handling)
6. [Examples](#examples)

## CLI Commands

### Global CLI Options

All CLI commands inherit these global options:

```python
@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file path")
@click.pass_context
def cli(ctx, verbose: bool, config: str | None):
    """Milvus Geo Search Benchmark Tool"""
```

**Parameters:**
- `verbose` (bool): Enable DEBUG level logging
- `config` (str, optional): Path to YAML configuration file

### generate-dataset

Generate synthetic geospatial datasets for benchmarking.

```python
@cli.command("generate-dataset")
@click.option("--num-points", default=100000, help="Number of training points")
@click.option("--num-queries", default=1000, help="Number of test queries")
@click.option("--output-dir", default="./data", help="Output directory")
@click.option("--bbox", default="-180,-90,180,90", help="Bounding box as min_lon,min_lat,max_lon,max_lat")
@click.option("--min-points-per-query", default=100, help="Minimum points per query")
@click.option("--max-radius", default=1.0, help="Maximum radius for queries")
@click.pass_context
def generate_dataset(ctx, num_points: int, num_queries: int, output_dir: str, 
                    bbox: str, min_points_per_query: int, max_radius: float):
```

**Parameters:**
- `num_points` (int): Number of training points to generate [1, 10^7]
- `num_queries` (int): Number of test queries to create [1, 10^5]
- `output_dir` (str): Directory path for output files
- `bbox` (str): Bounding box in format "min_lon,min_lat,max_lon,max_lat"
- `min_points_per_query` (int): Minimum points each query must match [1, 10^4]
- `max_radius` (float): Maximum radius for query polygons (degrees) [0.001, 10.0]

**Returns:**
- Creates three parquet files: train.parquet, test.parquet, ground_truth.parquet
- Prints summary of generated data

**Raises:**
- `click.BadParameter`: Invalid bbox format
- `ValueError`: Invalid parameter ranges
- `IOError`: Output directory creation failed

### load-data

Load training data into Milvus collection with proper schema and indexing.

```python
@cli.command("load-data")
@click.option("--uri", envvar="MILVUS_URI", required=True, help="Milvus URI")
@click.option("--token", envvar="MILVUS_TOKEN", required=True, help="Milvus token")
@click.option("--collection", default="geo_bench", help="Collection name")
@click.option("--data-file", required=True, type=click.Path(exists=True), help="Training data file")
@click.option("--batch-size", default=1000, help="Batch size for insertion")
@click.option("--recreate/--no-recreate", default=True, help="Recreate collection if exists")
@click.pass_context
def load_data(ctx, uri: str, token: str, collection: str, data_file: str, 
              batch_size: int, recreate: bool):
```

**Parameters:**
- `uri` (str): Milvus server URI (e.g., "http://localhost:19530")
- `token` (str): Authentication token (e.g., "root:Milvus")
- `collection` (str): Collection name for data storage
- `data_file` (str): Path to training data parquet file
- `batch_size` (int): Number of records per insertion batch [100, 10000]
- `recreate` (bool): Whether to drop and recreate existing collection

**Returns:**
- Prints collection statistics after successful loading

**Raises:**
- `ConnectionError`: Failed to connect to Milvus
- `FileNotFoundError`: Data file not found
- `MilvusException`: Collection or index creation failed

### run-benchmark

Execute performance benchmark tests on loaded collection.

```python
@cli.command("run-benchmark")
@click.option("--uri", envvar="MILVUS_URI", required=True, help="Milvus URI")
@click.option("--token", envvar="MILVUS_TOKEN", required=True, help="Milvus token")
@click.option("--collection", default="geo_bench", help="Collection name")
@click.option("--queries", required=True, type=click.Path(exists=True), help="Test queries file")
@click.option("--output", default="./data/results.parquet", help="Results output file")
@click.option("--timeout", default=30, help="Query timeout in seconds")
@click.option("--warmup", default=10, help="Number of warmup queries")
@click.pass_context
def run_benchmark(ctx, uri: str, token: str, collection: str, queries: str, 
                  output: str, timeout: int, warmup: int):
```

**Parameters:**
- `uri` (str): Milvus server URI
- `token` (str): Authentication token
- `collection` (str): Collection to query
- `queries` (str): Path to test queries parquet file
- `output` (str): Path for results parquet file
- `timeout` (int): Query timeout in seconds [1, 300]
- `warmup` (int): Number of warmup queries [0, 100]

**Returns:**
- Saves benchmark results to parquet file
- Prints performance summary

**Raises:**
- `ConnectionError`: Failed to connect to Milvus
- `CollectionNotExistsError`: Collection not found
- `TimeoutError`: Query timeout exceeded

### evaluate

Compare benchmark results against ground truth for accuracy evaluation.

```python
@cli.command("evaluate")
@click.option("--results", required=True, type=click.Path(exists=True), help="Benchmark results file")
@click.option("--ground-truth", required=True, type=click.Path(exists=True), help="Ground truth file")
@click.option("--output", default="./reports/evaluation_report.md", help="Evaluation report output")
@click.option("--format", "output_format", default="markdown", type=click.Choice(["markdown", "json"]), help="Output format")
@click.option("--print-summary/--no-print-summary", default=True, help="Print summary to console")
@click.pass_context
def evaluate(ctx, results: str, ground_truth: str, output: str, 
             output_format: str, print_summary: bool):
```

**Parameters:**
- `results` (str): Path to benchmark results parquet file
- `ground_truth` (str): Path to ground truth parquet file
- `output` (str): Path for evaluation report
- `output_format` (str): Report format ("markdown" or "json")
- `print_summary` (bool): Whether to print console summary

**Returns:**
- Generates evaluation report in specified format
- Optionally prints summary to console

**Raises:**
- `FileNotFoundError`: Input files not found
- `ValueError`: Incompatible file schemas
- `IOError`: Report generation failed

### full-run

Execute complete benchmark workflow from data generation to evaluation.

```python
@cli.command("full-run")
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file")
@click.option("--output-dir", default="./data", help="Output directory for datasets")
@click.option("--reports-dir", default="./reports", help="Output directory for reports")
@click.pass_context
def full_run(ctx, config: str | None, output_dir: str, reports_dir: str):
```

**Parameters:**
- `config` (str, optional): Path to YAML configuration file
- `output_dir` (str): Directory for dataset files
- `reports_dir` (str): Directory for evaluation reports

**Returns:**
- Executes complete pipeline
- Generates all output files and reports

**Raises:**
- `ConfigurationError`: Invalid configuration
- `WorkflowError`: Pipeline step failed

## Core Modules

### dataset.py

#### DatasetGenerator

Main class for generating synthetic geospatial datasets.

```python
class DatasetGenerator:
    """Generate datasets for geo search benchmarks."""
    
    def __init__(self, config: dict[str, Any]):
        """Initialize dataset generator with configuration.
        
        Args:
            config: Configuration dictionary with dataset parameters
            
        Raises:
            ValueError: Invalid configuration parameters
        """
```

**Configuration Schema:**
```python
{
    "dataset": {
        "num_points": int,           # Number of training points
        "num_queries": int,          # Number of test queries  
        "bbox": list[float],         # [min_lon, min_lat, max_lon, max_lat]
        "min_points_per_query": int, # Minimum results per query
        "max_radius": float          # Maximum query radius
    }
}
```

**Methods:**

##### generate_train_data()
```python
def generate_train_data(self, num_points: int) -> pd.DataFrame:
    """Generate training data with points and vectors.
    
    Args:
        num_points: Number of points to generate
        
    Returns:
        DataFrame with columns: id, wkt, vec
        
    Raises:
        ValueError: Invalid num_points parameter
    """
```

##### generate_test_queries()
```python
def generate_test_queries(self, train_df: pd.DataFrame, num_queries: int) -> pd.DataFrame:
    """Generate test queries that guarantee minimum number of results.
    
    Args:
        train_df: Training data DataFrame
        num_queries: Number of queries to generate
        
    Returns:
        DataFrame with columns: query_id, expr, polygon_wkt, center_lon, center_lat, radius
        
    Raises:
        ValueError: Insufficient training data for query generation
    """
```

##### calculate_ground_truth()
```python
def calculate_ground_truth(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ground truth results using Shapely.
    
    Args:
        train_df: Training data DataFrame
        test_df: Test queries DataFrame
        
    Returns:
        DataFrame with columns: query_id, result_ids, result_count
        
    Raises:
        GeometryError: Invalid WKT geometry
    """
```

##### generate_full_dataset()
```python
def generate_full_dataset(self, output_dir: str) -> dict[str, str]:
    """Generate complete dataset and save to parquet files.
    
    Args:
        output_dir: Directory for output files
        
    Returns:
        Dictionary mapping dataset type to file path
        
    Raises:
        IOError: File creation failed
    """
```

### milvus_client.py

#### MilvusGeoClient

Client wrapper for Milvus geo search operations.

```python
class MilvusGeoClient:
    """Milvus client for geo search operations."""
    
    def __init__(self, uri: str, token: str):
        """Initialize Milvus client with URI and token.
        
        Args:
            uri: Milvus server URI
            token: Authentication token
            
        Raises:
            ConnectionError: Failed to connect to Milvus
        """
```

**Methods:**

##### create_collection()
```python
def create_collection(self, collection_name: str, recreate: bool = True) -> None:
    """Create collection with geo and vector fields.
    
    Args:
        collection_name: Name for the collection
        recreate: Whether to drop existing collection
        
    Raises:
        MilvusException: Collection creation failed
    """
```

**Collection Schema:**
```python
schema = {
    "fields": [
        {"name": "id", "type": "INT64", "is_primary": True},
        {"name": "location", "type": "GEOMETRY"},
        {"name": "embedding", "type": "FLOAT_VECTOR", "dim": 8}
    ],
    "indexes": [
        {"field": "location", "type": "RTREE"},
        {"field": "embedding", "type": "IVF_FLAT", "metric": "L2", "params": {"nlist": 128}}
    ]
}
```

##### insert_data()
```python
def insert_data(self, collection_name: str, data_df: pd.DataFrame, batch_size: int = 1000) -> None:
    """Insert data from DataFrame into collection.
    
    Args:
        collection_name: Target collection name
        data_df: Training data DataFrame
        batch_size: Number of records per batch
        
    Raises:
        MilvusException: Data insertion failed
    """
```

##### search_geo()
```python
def search_geo(self, collection_name: str, expr: str, timeout: int = 30) -> tuple[list[int], float]:
    """Execute geo search query and return results with timing.
    
    Args:
        collection_name: Collection to search
        expr: Spatial query expression (ST_WITHIN format)
        timeout: Query timeout in seconds
        
    Returns:
        Tuple of (result_ids, query_time_ms)
        
    Raises:
        MilvusException: Query execution failed
        TimeoutError: Query timeout exceeded
    """
```

##### get_collection_stats()
```python
def get_collection_stats(self, collection_name: str) -> dict[str, Any]:
    """Get collection statistics.
    
    Args:
        collection_name: Collection name
        
    Returns:
        Dictionary with statistics (row_count, data_size, etc.)
        
    Raises:
        MilvusException: Failed to get statistics
    """
```

### benchmark.py

#### BenchmarkRunner

Class for executing performance benchmark tests.

```python
class BenchmarkRunner:
    """Execute benchmark tests against Milvus geo collections."""
    
    @classmethod
    def run_benchmark_from_config(cls, config: dict[str, Any], queries_file: str, 
                                  output_file: str) -> pd.DataFrame:
        """Run benchmark using configuration parameters.
        
        Args:
            config: Configuration dictionary
            queries_file: Path to test queries parquet file
            output_file: Path for results parquet file
            
        Returns:
            DataFrame with benchmark results
            
        Raises:
            ConnectionError: Failed to connect to Milvus
            FileNotFoundError: Queries file not found
        """
```

#### Benchmark

Individual benchmark execution class.

```python
class Benchmark:
    """Individual benchmark execution with detailed metrics."""
    
    def __init__(self, client: MilvusGeoClient, collection_name: str, timeout: int = 30):
        """Initialize benchmark with Milvus client.
        
        Args:
            client: Connected MilvusGeoClient instance
            collection_name: Collection to benchmark
            timeout: Query timeout in seconds
        """
    
    def run_queries(self, queries_df: pd.DataFrame, warmup: int = 10) -> pd.DataFrame:
        """Execute benchmark queries with warmup and measurement phases.
        
        Args:
            queries_df: Test queries DataFrame
            warmup: Number of warmup queries
            
        Returns:
            DataFrame with detailed results for each query
            
        Raises:
            MilvusException: Query execution failed
        """
```

### evaluator.py

#### Evaluator

Class for evaluating benchmark accuracy against ground truth.

```python
class Evaluator:
    """Evaluate benchmark results against ground truth."""
    
    def evaluate_results(self, results_file: str, ground_truth_file: str) -> dict[str, Any]:
        """Evaluate benchmark results against ground truth.
        
        Args:
            results_file: Path to benchmark results parquet file
            ground_truth_file: Path to ground truth parquet file
            
        Returns:
            Dictionary with comprehensive evaluation metrics
            
        Raises:
            FileNotFoundError: Input files not found
            ValueError: Incompatible schemas
        """
```

**Evaluation Metrics Schema:**
```python
{
    "summary": {
        "total_queries": int,
        "successful_queries": int,
        "evaluated_queries": int,
        "success_rate": float
    },
    "accuracy": {
        "macro_precision": float,     # [0.0, 1.0]
        "macro_recall": float,        # [0.0, 1.0] 
        "macro_f1": float,           # [0.0, 1.0]
        "micro_precision": float,     # [0.0, 1.0]
        "micro_recall": float,        # [0.0, 1.0]
        "micro_f1": float            # [0.0, 1.0]
    },
    "performance": {
        "query_time": {
            "mean": float,           # milliseconds
            "median": float,         # milliseconds
            "std": float,            # milliseconds
            "min": float,            # milliseconds
            "max": float,            # milliseconds
            "p95": float,            # milliseconds
            "p99": float             # milliseconds
        },
        "throughput": float,         # queries per second
        "result_count": {
            "mean": float,           # average results per query
            "median": float,
            "std": float,
            "min": int,
            "max": int
        }
    },
    "distributions": {
        "precision": dict,           # statistical distribution
        "recall": dict,              # statistical distribution  
        "f1_score": dict            # statistical distribution
    },
    "confusion_matrix": {
        "true_positives": int,
        "false_positives": int,
        "false_negatives": int
    },
    "per_query_metrics": list[dict] # detailed per-query breakdown
}
```

##### generate_report()
```python
def generate_report(self, metrics: dict[str, Any], output_format: str, 
                   output_file: str) -> None:
    """Generate evaluation report in specified format.
    
    Args:
        metrics: Evaluation metrics dictionary
        output_format: Report format ("markdown" or "json")
        output_file: Output file path
        
    Raises:
        ValueError: Invalid output format
        IOError: File writing failed
    """
```

##### print_summary()
```python
def print_summary(self, metrics: dict[str, Any]) -> None:
    """Print evaluation summary to console.
    
    Args:
        metrics: Evaluation metrics dictionary
    """
```

### utils.py

Utility functions for configuration, file I/O, and data processing.

#### Configuration Management

```python
def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load configuration from YAML file with environment variable substitution.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary with resolved environment variables
        
    Raises:
        FileNotFoundError: Configuration file not found
        yaml.YAMLError: Invalid YAML syntax
    """

def get_default_config() -> dict[str, Any]:
    """Get default configuration dictionary.
    
    Returns:
        Default configuration with standard parameters
    """
```

#### File Operations

```python
def save_parquet(df: pd.DataFrame, file_path: str | Path) -> None:
    """Save DataFrame to Parquet file with automatic directory creation.
    
    Args:
        df: DataFrame to save
        file_path: Output file path
        
    Raises:
        IOError: File writing failed
    """

def load_parquet(file_path: str | Path) -> pd.DataFrame:
    """Load DataFrame from Parquet file.
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: File not found
        ParquetReadError: Invalid parquet file
    """
```

#### Validation Functions

```python
def validate_wkt_point(wkt: str) -> bool:
    """Validate WKT Point format.
    
    Args:
        wkt: WKT string to validate
        
    Returns:
        True if valid Point WKT format
    """

def validate_wkt_polygon(wkt: str) -> bool:
    """Validate WKT Polygon format.
    
    Args:
        wkt: WKT string to validate
        
    Returns:
        True if valid Polygon WKT format
    """
```

## Data Types

### Training Data Record

```python
@dataclass
class TrainingRecord:
    """Training data record structure."""
    id: int                    # Unique identifier
    wkt: str                  # Point geometry in WKT format
    vec: list[float]          # 8-dimensional normalized vector
```

### Test Query Record

```python
@dataclass
class TestQuery:
    """Test query record structure."""
    query_id: int             # Query identifier
    expr: str                 # ST_WITHIN spatial expression
    polygon_wkt: str          # Query polygon in WKT format
    center_lon: float         # Query center longitude
    center_lat: float         # Query center latitude
    radius: float             # Query radius in degrees
```

### Benchmark Result Record

```python
@dataclass  
class BenchmarkResult:
    """Benchmark result record structure."""
    query_id: int             # Query identifier
    query_time_ms: float      # Execution time in milliseconds
    result_ids: list[int]     # Matching point IDs
    result_count: int         # Number of results
    success: bool             # Query success status
    error_message: str | None # Error details if failed
```

### Ground Truth Record

```python
@dataclass
class GroundTruthRecord:
    """Ground truth record structure."""
    query_id: int             # Query identifier
    result_ids: list[int]     # Expected matching point IDs
    result_count: int         # Number of expected results
```

## Configuration Schema

### Complete Configuration Structure

```yaml
# Dataset generation parameters
dataset:
  num_points: int           # Number of training points [1, 10^7]
  num_queries: int          # Number of test queries [1, 10^5]
  output_dir: str           # Output directory path
  bbox: list[float]         # Bounding box [min_lon, min_lat, max_lon, max_lat]
  min_points_per_query: int # Minimum points per query [1, 10^4]
  max_radius: float         # Maximum query radius [0.001, 10.0]

# Milvus connection parameters  
milvus:
  uri: str                  # Milvus server URI
  token: str                # Authentication token
  collection: str           # Collection name
  batch_size: int           # Insertion batch size [100, 10000]
  timeout: int              # Connection timeout [1, 300]

# Benchmark execution parameters
benchmark:
  timeout: int              # Query timeout [1, 300]
  warmup: int               # Number of warmup queries [0, 100]

# Output configuration
output:
  results: str              # Results file path
  report: str               # Report file path
```

### Environment Variable Substitution

Configuration files support environment variable substitution:

```yaml
milvus:
  uri: "${MILVUS_URI}"                    # Required
  token: "${MILVUS_TOKEN}"                # Required
  collection: "${MILVUS_COLLECTION:geo_bench}" # Optional with default
```

**Supported Formats:**
- `${VAR}` - Required variable (fails if not set)
- `${VAR:default}` - Optional variable with default value
- `${VAR:-default}` - Optional variable with default if empty

## Error Handling

### Exception Hierarchy

```python
class MilvusGeoBenchError(Exception):
    """Base exception for all tool-specific errors."""
    pass

class ConfigurationError(MilvusGeoBenchError):
    """Configuration-related errors."""
    pass

class DatasetGenerationError(MilvusGeoBenchError):
    """Dataset generation errors."""  
    pass

class MilvusConnectionError(MilvusGeoBenchError):
    """Milvus connection and operation errors."""
    pass

class BenchmarkExecutionError(MilvusGeoBenchError):
    """Benchmark execution errors."""
    pass

class EvaluationError(MilvusGeoBenchError):
    """Result evaluation errors."""
    pass
```

### Common Error Scenarios

#### Configuration Errors

```python
# Missing required environment variables
ConfigurationError("MILVUS_URI environment variable not set")

# Invalid parameter ranges  
ValueError("num_points must be between 1 and 10000000")

# Invalid bounding box format
click.BadParameter("bbox must be in format: min_lon,min_lat,max_lon,max_lat")
```

#### Connection Errors

```python
# Milvus server unreachable
ConnectionError("Failed to connect to Milvus at http://localhost:19530")

# Authentication failed
MilvusException("Authentication failed: invalid credentials")

# Collection operations failed
MilvusException("Collection 'geo_bench' does not exist")
```

#### Data Errors

```python
# File not found
FileNotFoundError("Training data file not found: ./data/train.parquet")

# Invalid data format
ValueError("Invalid WKT geometry: POINT(invalid coordinates)")

# Schema mismatch
ValueError("Results file missing required column: result_ids")
```

#### Runtime Errors

```python
# Query timeout
TimeoutError("Query timeout exceeded: 30 seconds")

# Resource exhaustion
MemoryError("Insufficient memory for batch size: 10000")

# Index creation failed
MilvusException("Failed to create RTREE index: unsupported geometry type")
```

### Error Handling Best Practices

#### Graceful Degradation

```python
# Example: Handle index creation failure gracefully
try:
    self._create_geometry_index(collection_name)
    logging.info("Created RTREE geometry index")
except MilvusException as e:
    logging.warning(f"Geometry index creation failed: {e}")
    logging.warning("Continuing without geometry index (reduced performance)")
```

#### Resource Cleanup

```python
# Example: Context manager for automatic cleanup
class MilvusGeoClient:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.close()
            logging.info("Closed Milvus connection")
```

#### Retry Logic

```python
# Example: Retry connection with exponential backoff
def _connect_with_retry(self, max_retries: int = 3) -> None:
    for attempt in range(max_retries):
        try:
            self.client = MilvusClient(uri=self.uri, token=self.token)
            return
        except ConnectionError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt
            logging.warning(f"Connection attempt {attempt + 1} failed, retrying in {wait_time}s")
            time.sleep(wait_time)
```

## Examples

### Programmatic Usage

#### Custom Dataset Generation

```python
from milvus_geo_bench.dataset import DatasetGenerator

# Create custom configuration
config = {
    "dataset": {
        "num_points": 50000,
        "num_queries": 500,
        "bbox": [-122.5, 37.7, -122.3, 37.9],  # San Francisco Bay Area
        "min_points_per_query": 100,
        "max_radius": 0.01
    }
}

# Generate dataset
generator = DatasetGenerator(config)
files = generator.generate_full_dataset("./sf_benchmark")

print(f"Generated files: {files}")
# Output: {'train': './sf_benchmark/train.parquet', 'test': './sf_benchmark/test.parquet', 'ground_truth': './sf_benchmark/ground_truth.parquet'}
```

#### Direct Milvus Operations

```python
from milvus_geo_bench.milvus_client import MilvusGeoClient
import pandas as pd

# Connect to Milvus
with MilvusGeoClient("http://localhost:19530", "root:Milvus") as client:
    # Create collection
    client.create_collection("custom_geo_test")
    
    # Load data
    train_df = pd.read_parquet("./data/train.parquet")
    client.insert_data("custom_geo_test", train_df, batch_size=2000)
    
    # Execute single query
    spatial_expr = "ST_WITHIN(location, 'POLYGON((-1 -1, 1 -1, 1 1, -1 1, -1 -1))')"
    result_ids, query_time = client.search_geo("custom_geo_test", spatial_expr)
    
    print(f"Found {len(result_ids)} results in {query_time:.2f}ms")
```

#### Custom Benchmark Execution

```python
from milvus_geo_bench.benchmark import Benchmark
from milvus_geo_bench.milvus_client import MilvusGeoClient
import pandas as pd

# Load test queries
queries_df = pd.read_parquet("./data/test.parquet")

# Run benchmark with custom parameters
with MilvusGeoClient("http://localhost:19530", "root:Milvus") as client:
    benchmark = Benchmark(client, "geo_test", timeout=60)
    results_df = benchmark.run_queries(queries_df, warmup=20)
    
    # Analyze results
    success_rate = results_df["success"].mean()
    avg_time = results_df[results_df["success"]]["query_time_ms"].mean()
    
    print(f"Success rate: {success_rate:.2%}")
    print(f"Average query time: {avg_time:.2f}ms")
```

#### Custom Evaluation

```python
from milvus_geo_bench.evaluator import Evaluator

# Evaluate results
evaluator = Evaluator()
metrics = evaluator.evaluate_results("./data/results.parquet", "./data/ground_truth.parquet")

# Extract specific metrics
precision = metrics["accuracy"]["macro_precision"]
recall = metrics["accuracy"]["macro_recall"]
f1_score = metrics["accuracy"]["macro_f1"]
avg_query_time = metrics["performance"]["query_time"]["mean"]

print(f"Accuracy: P={precision:.3f}, R={recall:.3f}, F1={f1_score:.3f}")
print(f"Performance: {avg_query_time:.2f}ms avg query time")

# Generate custom report
evaluator.generate_report(metrics, "json", "./custom_metrics.json")
```

This API reference provides complete documentation for programmatic usage of the Milvus Geo Benchmark CLI tool. All functions include proper type hints, error handling, and comprehensive parameter validation.