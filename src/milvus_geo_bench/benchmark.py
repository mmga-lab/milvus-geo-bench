"""
Benchmark execution module for Milvus geo search.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from typing import Any

import pandas as pd
from tqdm import tqdm

from .milvus_client import MilvusGeoClient
from .utils import load_parquet, save_parquet


class Benchmark:
    """Execute geo search benchmarks against Milvus."""

    def __init__(self, config: dict[str, Any]):
        self.config = config["benchmark"]
        self.milvus_config = config["milvus"]
        self.timeout = self.config.get("timeout", 30)
        self.warmup_queries = self.config.get("warmup", 10)
        self.concurrency = self.config.get("concurrency", 1)

        logging.info(
            f"Benchmark initialized with timeout: {self.timeout}s, warmup: {self.warmup_queries}, concurrency: {self.concurrency}"
        )

    def run_full_benchmark(
        self, client: MilvusGeoClient, collection_name: str, queries_file: str, output_file: str
    ) -> pd.DataFrame:
        """Run complete benchmark including warmup and measurement."""

        # Load test queries
        queries_df = load_parquet(queries_file)
        logging.info(f"Loaded {len(queries_df)} test queries")

        # Health check
        if not client.health_check():
            raise RuntimeError("Milvus health check failed")

        # Verify collection and indexes are ready
        self._verify_collection_ready(client, collection_name)

        # Warmup phase
        self._run_warmup(client, collection_name, queries_df)

        # Main benchmark
        results_df = self._run_benchmark(client, collection_name, queries_df)

        # Save results
        save_parquet(results_df, output_file)

        # Log summary statistics
        self._log_benchmark_summary(results_df)

        return results_df

    def _verify_collection_ready(self, client: MilvusGeoClient, collection_name: str) -> None:
        """Verify that collection and all indexes are ready for benchmarking."""
        logging.info(f"Verifying collection '{collection_name}' readiness...")

        try:
            # Check collection statistics
            stats = client.get_collection_stats(collection_name)
            if stats:
                row_count = stats.get("row_count", 0)
                logging.info(f"Collection has {row_count} rows")

                if row_count == 0:
                    logging.warning("Collection appears to be empty")

            # Check index status for critical fields
            fields_to_check = ["embedding", "location"]
            for field_name in fields_to_check:
                index_info = client.get_index_status(collection_name, field_name)
                if index_info:
                    state = index_info.get("state", "Unknown")
                    index_type = index_info.get("index_type", "Unknown")
                    total_rows = index_info.get("total_rows", 0)
                    indexed_rows = index_info.get("indexed_rows", 0)
                    pending_rows = index_info.get("pending_index_rows", 0)

                    logging.info(
                        f"Index for '{field_name}': type={index_type}, state={state}, "
                        f"progress={indexed_rows}/{total_rows}, pending={pending_rows}"
                    )

                    if not client.is_index_ready(collection_name, field_name):
                        logging.warning(f"Index for field '{field_name}' is still building")
                else:
                    logging.warning(f"No index found for field '{field_name}'")

            # Final readiness check with brief wait
            logging.info("Performing final index readiness check...")
            if client.wait_for_indexes_ready(collection_name, timeout=30, check_interval=2):
                logging.info("Collection verification completed - all systems ready")
            else:
                logging.warning("Some indexes may still be building, but proceeding with benchmark")

        except Exception as e:
            logging.error(f"Collection readiness verification failed: {e}")
            raise RuntimeError(
                f"Collection '{collection_name}' is not ready for benchmarking: {e}"
            ) from e

    def _run_warmup(
        self, client: MilvusGeoClient, collection_name: str, queries_df: pd.DataFrame
    ) -> None:
        """Run warmup queries to prepare the system."""
        if self.warmup_queries <= 0:
            logging.info("Skipping warmup phase")
            return

        logging.info(f"Starting warmup with {self.warmup_queries} queries...")

        # Select random queries for warmup
        warmup_queries = queries_df.sample(n=min(self.warmup_queries, len(queries_df)))

        warmup_times = []
        for _, query_row in tqdm(
            warmup_queries.iterrows(), total=len(warmup_queries), desc="Warmup queries"
        ):
            try:
                _, query_time = client.search_geo(
                    collection_name=collection_name, expr=query_row["expr"], timeout=self.timeout
                )
                warmup_times.append(query_time)
            except Exception as e:
                logging.warning(f"Warmup query {query_row['query_id']} failed: {e}")

        avg_warmup_time = sum(warmup_times) / len(warmup_times) if warmup_times else 0
        logging.info(f"Warmup completed. Average query time: {avg_warmup_time:.2f}ms")

    def _run_benchmark(
        self, client: MilvusGeoClient, collection_name: str, queries_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Run the main benchmark queries."""
        logging.info(
            f"Starting benchmark with {len(queries_df)} queries (concurrency: {self.concurrency})..."
        )

        if self.concurrency == 1:
            return self._run_benchmark_serial(client, collection_name, queries_df)
        else:
            return self._run_benchmark_concurrent(client, collection_name, queries_df)

    def _run_benchmark_serial(
        self, client: MilvusGeoClient, collection_name: str, queries_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Run benchmark queries serially (original implementation)."""
        results = []
        failed_queries = 0

        for _, query_row in tqdm(
            queries_df.iterrows(), total=len(queries_df), desc="Benchmark queries (serial)"
        ):
            query_id = query_row["query_id"]
            expr = query_row["expr"]

            try:
                # Execute query
                result_ids, query_time = client.search_geo(
                    collection_name=collection_name, expr=expr, timeout=self.timeout
                )

                # Record result
                results.append(
                    {
                        "query_id": query_id,
                        "query_time_ms": query_time,
                        "result_ids": result_ids,
                        "result_count": len(result_ids),
                        "success": True,
                        "error_message": None,
                    }
                )

            except Exception as e:
                logging.error(f"Query {query_id} failed: {e}")
                failed_queries += 1

                # Record failed query
                results.append(
                    {
                        "query_id": query_id,
                        "query_time_ms": None,
                        "result_ids": [],
                        "result_count": 0,
                        "success": False,
                        "error_message": str(e),
                    }
                )

        if failed_queries > 0:
            logging.warning(f"{failed_queries} queries failed out of {len(queries_df)}")

        return pd.DataFrame(results)

    def _run_benchmark_concurrent(
        self, client: MilvusGeoClient, collection_name: str, queries_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Run benchmark queries concurrently using ThreadPoolExecutor."""
        results = []
        failed_queries = 0

        def execute_single_query(query_row):
            """Execute a single query and return result."""
            query_id = query_row["query_id"]
            expr = query_row["expr"]

            try:
                result_ids, query_time = client.search_geo(
                    collection_name=collection_name, expr=expr, timeout=self.timeout
                )

                return {
                    "query_id": query_id,
                    "query_time_ms": query_time,
                    "result_ids": result_ids,
                    "result_count": len(result_ids),
                    "success": True,
                    "error_message": None,
                }

            except Exception as e:
                return {
                    "query_id": query_id,
                    "query_time_ms": None,
                    "result_ids": [],
                    "result_count": 0,
                    "success": False,
                    "error_message": str(e),
                }

        # Record total execution time
        total_start_time = time.time()

        # Execute queries concurrently
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            # Submit all queries
            future_to_query = {
                executor.submit(execute_single_query, query_row): query_row
                for _, query_row in queries_df.iterrows()
            }

            # Collect results with progress bar
            with tqdm(total=len(queries_df), desc="Benchmark queries (concurrent)") as pbar:
                for future in as_completed(future_to_query):
                    result = future.result()
                    results.append(result)

                    if not result["success"]:
                        failed_queries += 1
                        logging.error(
                            f"Query {result['query_id']} failed: {result['error_message']}"
                        )

                    pbar.update(1)

        total_end_time = time.time()
        total_execution_time = total_end_time - total_start_time

        # Add total execution time to results for throughput calculation
        for result in results:
            result["total_execution_time_s"] = total_execution_time

        if failed_queries > 0:
            logging.warning(f"{failed_queries} queries failed out of {len(queries_df)}")

        return pd.DataFrame(results)

    def _log_benchmark_summary(self, results_df: pd.DataFrame) -> None:
        """Log summary statistics of the benchmark."""
        successful_results = results_df[results_df["success"]]

        if len(successful_results) == 0:
            logging.error("No successful queries to analyze")
            return

        query_times = successful_results["query_time_ms"].values
        result_counts = successful_results["result_count"].values

        # Query time statistics
        logging.info("=== Benchmark Summary ===")
        logging.info(f"Total queries: {len(results_df)}")
        logging.info(f"Successful queries: {len(successful_results)}")
        logging.info(f"Failed queries: {len(results_df) - len(successful_results)}")
        logging.info(f"Success rate: {len(successful_results) / len(results_df) * 100:.2f}%")

        logging.info("Query time statistics (ms):")
        logging.info(f"  Mean: {query_times.mean():.2f}")
        logging.info(f"  Median: {pd.Series(query_times).median():.2f}")
        logging.info(f"  Min: {query_times.min():.2f}")
        logging.info(f"  Max: {query_times.max():.2f}")
        logging.info(f"  P95: {pd.Series(query_times).quantile(0.95):.2f}")
        logging.info(f"  P99: {pd.Series(query_times).quantile(0.99):.2f}")

        logging.info("Result count statistics:")
        logging.info(f"  Mean: {result_counts.mean():.2f}")
        logging.info(f"  Median: {pd.Series(result_counts).median():.2f}")
        logging.info(f"  Min: {result_counts.min()}")
        logging.info(f"  Max: {result_counts.max()}")

        # Throughput calculation
        if self.concurrency > 1 and "total_execution_time_s" in successful_results.columns:
            # For concurrent execution, use actual total execution time
            total_execution_time = successful_results["total_execution_time_s"].iloc[0]
            throughput = (
                len(successful_results) / total_execution_time if total_execution_time > 0 else 0
            )
            logging.info(f"Total execution time: {total_execution_time:.2f}s")
            logging.info(f"Throughput (concurrent): {throughput:.2f} queries/second")
        else:
            # Serial execution: sum all query times
            total_time = query_times.sum() / 1000  # Convert to seconds
            throughput = len(successful_results) / total_time if total_time > 0 else 0
            logging.info(f"Throughput (serial): {throughput:.2f} queries/second")

    def run_single_query(
        self, client: MilvusGeoClient, collection_name: str, expr: str
    ) -> tuple[list[int], float]:
        """Run a single query for testing purposes."""
        try:
            result_ids, query_time = client.search_geo(
                collection_name=collection_name, expr=expr, timeout=self.timeout
            )

            logging.info(
                f"Query executed in {query_time:.2f}ms, returned {len(result_ids)} results"
            )
            return result_ids, query_time

        except Exception as e:
            logging.error(f"Single query failed: {e}")
            raise


class BenchmarkRunner:
    """High-level interface for running benchmarks."""

    @staticmethod
    def run_benchmark_from_config(
        config: dict[str, Any], queries_file: str, output_file: str
    ) -> pd.DataFrame:
        """Run benchmark using configuration."""

        # Create Milvus client
        milvus_config = config["milvus"]
        with MilvusGeoClient(uri=milvus_config["uri"], token=milvus_config["token"]) as client:
            # Create benchmark instance
            benchmark = Benchmark(config)

            # Run benchmark
            results_df = benchmark.run_full_benchmark(
                client=client,
                collection_name=milvus_config["collection"],
                queries_file=queries_file,
                output_file=output_file,
            )

            return results_df
