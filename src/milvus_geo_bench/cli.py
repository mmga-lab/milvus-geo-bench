from pathlib import Path

import click

from .benchmark import BenchmarkRunner
from .dataset import DatasetGenerator
from .evaluator import Evaluator
from .milvus_client import MilvusGeoClient
from .utils import ensure_dir, load_config, load_parquet, load_train_data, setup_logging


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file path")
@click.pass_context
def cli(ctx, verbose: bool, config: str | None):
    """Milvus Geo Search Benchmark Tool"""
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)

    # Load configuration
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)
    ctx.obj["verbose"] = verbose


@cli.command("generate-dataset")
@click.option("--num-points", type=int, help="Number of training points")
@click.option("--num-queries", type=int, help="Number of test queries")
@click.option("--output-dir", help="Output directory")
@click.option("--bbox", help="Bounding box as min_lon,min_lat,max_lon,max_lat")
@click.option("--min-points-per-query", type=int, help="Minimum points per query")
@click.option(
    "--grid-enabled", is_flag=True, help="Enable grid-based generation for large datasets"
)
@click.option("--num-grids", type=int, help="Number of grids (overrides auto-calculation)")
@click.option(
    "--target-points-per-grid", type=int, help="Target points per grid for auto-calculation"
)
@click.pass_context
def generate_dataset(
    ctx,
    num_points: int | None,
    num_queries: int | None,
    output_dir: str | None,
    bbox: str | None,
    min_points_per_query: int | None,
    grid_enabled: bool,
    num_grids: int | None,
    target_points_per_grid: int | None,
):
    """Generate training and test datasets"""

    config = ctx.obj["config"].copy()
    dataset_config = config["dataset"]

    # Override with command line arguments if provided
    if num_points is not None:
        dataset_config["num_points"] = num_points
    if num_queries is not None:
        dataset_config["num_queries"] = num_queries
    if output_dir is not None:
        dataset_config["output_dir"] = output_dir
    if min_points_per_query is not None:
        dataset_config["min_points_per_query"] = min_points_per_query
    if bbox is not None:
        bbox_list = list(map(float, bbox.split(",")))
        if len(bbox_list) != 4:
            raise click.BadParameter("bbox must be in format: min_lon,min_lat,max_lon,max_lat")
        dataset_config["bbox"] = bbox_list

    # Grid configuration overrides
    if grid_enabled:
        dataset_config["grid"]["enabled"] = True
    if num_grids is not None:
        dataset_config["grid"]["enabled"] = True
        dataset_config["grid"]["auto_calculate"] = False
        dataset_config["grid"]["num_grids"] = num_grids
    if target_points_per_grid is not None:
        dataset_config["grid"]["enabled"] = True
        dataset_config["grid"]["target_points_per_grid"] = target_points_per_grid

    # Auto-enable grid for large datasets
    if dataset_config["num_points"] >= 10_000_000 and not dataset_config["grid"]["enabled"]:
        click.echo("Auto-enabling grid mode for large dataset (>= 10M points)")
        dataset_config["grid"]["enabled"] = True

    # Get final values
    final_output_dir = dataset_config["output_dir"]

    click.echo(
        f"Generating dataset with {dataset_config['num_points']} points and {dataset_config['num_queries']} queries"
    )
    click.echo(f"Output directory: {final_output_dir}")
    click.echo(f"Bounding box: {dataset_config['bbox']}")

    # Generate dataset
    generator = DatasetGenerator(config)
    files = generator.generate_full_dataset(final_output_dir)

    click.echo("Dataset generation completed:")
    for dataset_type, file_path in files.items():
        click.echo(f"  {dataset_type}: {file_path}")


@cli.command("load-data")
@click.option("--uri", envvar="MILVUS_URI", help="Milvus URI")
@click.option("--token", envvar="MILVUS_TOKEN", help="Milvus token")
@click.option("--collection", help="Collection name")
@click.option("--data-file", type=click.Path(exists=True), help="Training data file")
@click.option("--batch-size", type=int, help="Batch size for insertion")
@click.option(
    "--recreate/--no-recreate", default=True, help="Recreate collection if exists (default: True)"
)
@click.pass_context
def load_data(
    ctx,
    uri: str | None,
    token: str | None,
    collection: str | None,
    data_file: str | None,
    batch_size: int | None,
    recreate: bool,
):
    """Load data into Milvus collection"""

    config = ctx.obj["config"].copy()
    milvus_config = config["milvus"]

    # Override with command line arguments if provided
    if uri is not None:
        milvus_config["uri"] = uri
    if token is not None:
        milvus_config["token"] = token
    if collection is not None:
        milvus_config["collection"] = collection
    if batch_size is not None:
        milvus_config["batch_size"] = batch_size
    if data_file is None:
        # Default to train directory or train.parquet
        data_file = "./data/train" if Path("./data/train").exists() else "./data/train.parquet"

    # Validate required parameters
    if not milvus_config.get("uri"):
        raise click.UsageError(
            "Milvus URI is required. Set MILVUS_URI environment variable or use --uri option."
        )
    if not milvus_config.get("token"):
        raise click.UsageError(
            "Milvus token is required. Set MILVUS_TOKEN environment variable or use --token option."
        )

    # Get final values
    final_uri = milvus_config["uri"]
    final_token = milvus_config["token"]
    final_collection = milvus_config["collection"]
    final_batch_size = milvus_config["batch_size"]

    click.echo(f"Loading data from {data_file} to Milvus collection '{final_collection}'...")
    click.echo(f"Batch size: {final_batch_size}, Recreate: {recreate}")

    # Load data (can handle both single files and directories)
    train_df = load_train_data(data_file)

    # Connect to Milvus and load data
    with MilvusGeoClient(uri=final_uri, token=final_token) as client:
        # Create collection
        client.create_collection(final_collection, recreate=recreate)

        # Insert data
        client.insert_data(final_collection, train_df, final_batch_size)

        # Get stats
        stats = client.get_collection_stats(final_collection)
        click.echo(f"Data loading completed. Collection stats: {stats}")


@cli.command("run-benchmark")
@click.option("--uri", envvar="MILVUS_URI", help="Milvus URI")
@click.option("--token", envvar="MILVUS_TOKEN", help="Milvus token")
@click.option("--collection", help="Collection name")
@click.option("--queries", type=click.Path(exists=True), help="Test queries file")
@click.option("--output", help="Results output file")
@click.option("--timeout", type=int, help="Query timeout in seconds")
@click.option("--warmup", type=int, help="Number of warmup queries")
@click.option("--concurrency", type=int, help="Number of concurrent threads for query execution")
@click.pass_context
def run_benchmark(
    ctx,
    uri: str | None,
    token: str | None,
    collection: str | None,
    queries: str | None,
    output: str | None,
    timeout: int | None,
    warmup: int | None,
    concurrency: int | None,
):
    """Execute benchmark tests"""

    config = ctx.obj["config"].copy()

    # Use config defaults, override with command line arguments if provided
    milvus_config = config["milvus"]
    benchmark_config = config["benchmark"]
    output_config = config["output"]

    # Override with command line arguments if provided
    if uri is not None:
        milvus_config["uri"] = uri
    if token is not None:
        milvus_config["token"] = token
    if collection is not None:
        milvus_config["collection"] = collection
    if timeout is not None:
        benchmark_config["timeout"] = timeout
    if warmup is not None:
        benchmark_config["warmup"] = warmup
    if concurrency is not None:
        benchmark_config["concurrency"] = concurrency
    if output is not None:
        output_config["results"] = output
    if queries is None:
        queries = "./data/test.parquet"  # Default queries file

    # Validate required parameters
    if not milvus_config.get("uri"):
        raise click.UsageError(
            "Milvus URI is required. Set MILVUS_URI environment variable or use --uri option."
        )
    if not milvus_config.get("token"):
        raise click.UsageError(
            "Milvus token is required. Set MILVUS_TOKEN environment variable or use --token option."
        )

    # Get final values
    final_collection = milvus_config["collection"]
    final_output = output_config["results"]
    final_timeout = benchmark_config["timeout"]
    final_warmup = benchmark_config["warmup"]
    final_concurrency = benchmark_config["concurrency"]

    click.echo(f"Running benchmark on collection '{final_collection}'...")
    click.echo(f"Using queries from: {queries}")
    click.echo(f"Output file: {final_output}")
    click.echo(
        f"Timeout: {final_timeout}s, Warmup: {final_warmup}, Concurrency: {final_concurrency}"
    )

    # Ensure output directory exists
    output_path = Path(final_output)
    ensure_dir(output_path.parent)

    # Run benchmark
    results_df = BenchmarkRunner.run_benchmark_from_config(
        config=config, queries_file=queries, output_file=final_output
    )

    click.echo(f"Benchmark completed. Results saved to {final_output}")
    click.echo(f"Total queries: {len(results_df)}")
    successful = len(results_df[results_df["success"]])
    click.echo(f"Successful queries: {successful}")
    click.echo(f"Success rate: {successful / len(results_df) * 100:.2f}%")


@cli.command("evaluate")
@click.option("--results", type=click.Path(exists=True), help="Benchmark results file")
@click.option("--ground-truth", type=click.Path(exists=True), help="Ground truth file")
@click.option("--output", help="Evaluation report output")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["markdown", "json"]),
    help="Output format",
)
@click.option("--print-summary/--no-print-summary", default=True, help="Print summary to console")
@click.pass_context
def evaluate(
    ctx,
    results: str | None,
    ground_truth: str | None,
    output: str | None,
    output_format: str | None,
    print_summary: bool | None,
):
    """Evaluate benchmark results"""

    config = ctx.obj["config"].copy()
    output_config = config["output"]

    # Use defaults from config, override with command line arguments if provided
    if results is None:
        results = "./data/results.parquet"  # Default results file
    if ground_truth is None:
        ground_truth = "./data/ground_truth.parquet"  # Default ground truth file
    if output is None:
        output = output_config["report"]
    if output_format is None:
        output_format = "markdown"  # Default format

    click.echo(f"Evaluating results from {results} against {ground_truth}...")
    click.echo(f"Output: {output}, Format: {output_format}, Print summary: {print_summary}")

    # Create evaluator and run evaluation
    evaluator = Evaluator()
    metrics = evaluator.evaluate_results(results, ground_truth)

    # Generate report
    ensure_dir(Path(output).parent)
    evaluator.generate_report(metrics=metrics, output_format=output_format, output_file=output)

    if print_summary:
        evaluator.print_summary(metrics)

    click.echo(f"Evaluation report saved to {output}")


@cli.command("full-run")
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file")
@click.option("--output-dir", default="./data", help="Output directory for datasets")
@click.option("--reports-dir", default="./reports", help="Output directory for reports")
@click.pass_context
def full_run(ctx, config: str | None, output_dir: str, reports_dir: str):
    """Execute complete benchmark workflow"""

    # Load configuration (override context config if provided)
    full_config = load_config(config) if config else ctx.obj["config"]

    click.echo("Starting full benchmark workflow...")

    # Step 1: Generate dataset
    click.echo("\n=== Step 1: Generating Dataset ===")
    generator = DatasetGenerator(full_config)
    files = generator.generate_full_dataset(output_dir)
    click.echo("Dataset generation completed")

    # Step 2: Load data
    click.echo("\n=== Step 2: Loading Data to Milvus ===")
    milvus_config = full_config["milvus"]

    if not milvus_config.get("uri") or not milvus_config.get("token"):
        click.echo("Error: Milvus URI and token must be configured")
        click.echo("Set MILVUS_URI and MILVUS_TOKEN environment variables or use config file")
        return

    train_df = load_parquet(files["train"])

    with MilvusGeoClient(uri=milvus_config["uri"], token=milvus_config["token"]) as client:
        client.create_collection(milvus_config["collection"], recreate=True)
        client.insert_data(milvus_config["collection"], train_df, milvus_config["batch_size"])

    click.echo("Data loading completed")

    # Step 3: Run benchmark
    click.echo("\n=== Step 3: Running Benchmark ===")
    results_file = f"{output_dir}/results.parquet"
    BenchmarkRunner.run_benchmark_from_config(
        config=full_config, queries_file=files["test"], output_file=results_file
    )
    click.echo("Benchmark execution completed")

    # Step 4: Evaluate results
    click.echo("\n=== Step 4: Evaluating Results ===")
    ensure_dir(reports_dir)
    evaluator = Evaluator()
    metrics = evaluator.evaluate_results(results_file, files["ground_truth"])

    # Generate both markdown and JSON reports
    md_report_file = f"{reports_dir}/evaluation_report.md"
    json_report_file = f"{reports_dir}/evaluation_metrics.json"

    evaluator.generate_report(metrics, "markdown", md_report_file)
    evaluator.generate_report(metrics, "json", json_report_file)

    # Print summary
    evaluator.print_summary(metrics)

    click.echo("\n=== Workflow Completed ===")
    click.echo("Generated files:")
    click.echo(f"  Dataset: {output_dir}/")
    click.echo(f"  Results: {results_file}")
    click.echo(f"  Reports: {reports_dir}/")


def main() -> None:
    """Main entry point for the CLI."""
    cli()
