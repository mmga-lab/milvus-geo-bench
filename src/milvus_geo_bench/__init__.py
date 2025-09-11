from pathlib import Path
from typing import Optional

import click

from .benchmark import BenchmarkRunner
from .dataset import DatasetGenerator
from .evaluator import Evaluator
from .milvus_client import MilvusGeoClient
from .utils import ensure_dir, load_config, setup_logging


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
@click.option("--num-points", default=100000, help="Number of training points")
@click.option("--num-queries", default=1000, help="Number of test queries")
@click.option("--output-dir", default="./data", help="Output directory")
@click.option(
    "--bbox", default="-180,-90,180,90", help="Bounding box as min_lon,min_lat,max_lon,max_lat"
)
@click.option("--min-points-per-query", default=100, help="Minimum points per query")
@click.pass_context
def generate_dataset(
    ctx,
    num_points: int,
    num_queries: int,
    output_dir: str,
    bbox: str,
    min_points_per_query: int,
):
    """Generate training and test datasets"""

    config = ctx.obj["config"].copy()

    # Update config with command line parameters
    bbox_list = list(map(float, bbox.split(",")))
    if len(bbox_list) != 4:
        raise click.BadParameter("bbox must be in format: min_lon,min_lat,max_lon,max_lat")

    config["dataset"].update(
        {
            "num_points": num_points,
            "num_queries": num_queries,
            "output_dir": output_dir,
            "bbox": bbox_list,
            "min_points_per_query": min_points_per_query,
        }
    )

    # Generate dataset
    generator = DatasetGenerator(config)
    files = generator.generate_full_dataset(output_dir)

    click.echo("Dataset generation completed:")
    for dataset_type, file_path in files.items():
        click.echo(f"  {dataset_type}: {file_path}")


@cli.command("load-data")
@click.option("--uri", envvar="MILVUS_URI", required=True, help="Milvus URI")
@click.option("--token", envvar="MILVUS_TOKEN", required=True, help="Milvus token")
@click.option("--collection", default="geo_bench", help="Collection name")
@click.option("--data-file", required=True, type=click.Path(exists=True), help="Training data file")
@click.option("--batch-size", default=1000, help="Batch size for insertion")
@click.option("--recreate/--no-recreate", default=True, help="Recreate collection if exists")
@click.pass_context
def load_data(
    _ctx, uri: str, token: str, collection: str, data_file: str, batch_size: int, recreate: bool
):
    """Load data into Milvus collection"""

    from .utils import load_parquet

    click.echo(f"Loading data from {data_file} to Milvus collection '{collection}'...")

    # Load data
    train_df = load_parquet(data_file)

    # Connect to Milvus and load data
    with MilvusGeoClient(uri=uri, token=token) as client:
        # Create collection
        client.create_collection(collection, recreate=recreate)

        # Insert data
        client.insert_data(collection, train_df, batch_size)

        # Get stats
        stats = client.get_collection_stats(collection)
        click.echo(f"Data loading completed. Collection stats: {stats}")


@cli.command("run-benchmark")
@click.option("--uri", envvar="MILVUS_URI", required=True, help="Milvus URI")
@click.option("--token", envvar="MILVUS_TOKEN", required=True, help="Milvus token")
@click.option("--collection", default="geo_bench", help="Collection name")
@click.option("--queries", required=True, type=click.Path(exists=True), help="Test queries file")
@click.option("--output", default="./data/results.parquet", help="Results output file")
@click.option("--timeout", default=30, help="Query timeout in seconds")
@click.option("--warmup", default=10, help="Number of warmup queries")
@click.option("--concurrency", default=1, help="Number of concurrent threads for query execution")
@click.pass_context
def run_benchmark(
    ctx,
    uri: str,
    token: str,
    collection: str,
    queries: str,
    output: str,
    timeout: int,
    warmup: int,
    concurrency: int,
):
    """Execute benchmark tests"""

    config = ctx.obj["config"].copy()

    # Update config
    config["milvus"].update({"uri": uri, "token": token, "collection": collection})
    config["benchmark"].update({"timeout": timeout, "warmup": warmup, "concurrency": concurrency})

    click.echo(f"Running benchmark on collection '{collection}'...")

    # Ensure output directory exists
    output_path = Path(output)
    ensure_dir(output_path.parent)

    # Run benchmark
    results_df = BenchmarkRunner.run_benchmark_from_config(
        config=config, queries_file=queries, output_file=output
    )

    click.echo(f"Benchmark completed. Results saved to {output}")
    click.echo(f"Total queries: {len(results_df)}")
    successful = len(results_df[results_df["success"]])
    click.echo(f"Successful queries: {successful}")
    click.echo(f"Success rate: {successful / len(results_df) * 100:.2f}%")


@cli.command("evaluate")
@click.option(
    "--results", required=True, type=click.Path(exists=True), help="Benchmark results file"
)
@click.option(
    "--ground-truth", required=True, type=click.Path(exists=True), help="Ground truth file"
)
@click.option("--output", default="./reports/evaluation_report.md", help="Evaluation report output")
@click.option(
    "--format",
    "output_format",
    default="markdown",
    type=click.Choice(["markdown", "json"]),
    help="Output format",
)
@click.option("--print-summary/--no-print-summary", default=True, help="Print summary to console")
@click.pass_context
def evaluate(
    ctx, results: str, ground_truth: str, output: str, output_format: str, print_summary: bool
):
    """Evaluate benchmark results"""

    click.echo(f"Evaluating results from {results} against {ground_truth}...")

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

    from .utils import load_parquet

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
