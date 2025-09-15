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

        # Support both single concurrency and list of concurrency values
        concurrency_config = self.config.get("concurrency", 1)
        if isinstance(concurrency_config, list):
            self.concurrency_levels = concurrency_config
        else:
            self.concurrency_levels = [concurrency_config]

        # Auto-scaling configuration
        self.auto_extend_concurrency = self.config.get("auto_extend_concurrency", False)
        self.max_concurrency_limit = self.config.get("max_concurrency_limit", 1000)
        self.qps_growth_threshold = self.config.get("qps_growth_threshold", 5)
        self.max_error_rate = self.config.get("max_error_rate", 0.05)

        # Index creation configuration
        self.create_vector_index = self.config.get("create_vector_index", True)
        self.create_geo_index = self.config.get("create_geo_index", True)


        logging.info(
            f"Benchmark initialized with timeout: {self.timeout}s, warmup: {self.warmup_queries}"
        )
        logging.info(f"Concurrency levels: {self.concurrency_levels}")
        logging.info(f"Index creation - Vector: {self.create_vector_index}, Geo: {self.create_geo_index}")
        if self.auto_extend_concurrency:
            logging.info(f"Auto-scaling enabled - Max limit: {self.max_concurrency_limit}, Growth threshold: {self.qps_growth_threshold}%")

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

        # Release other collections to free up memory for benchmark
        release_times = self._release_other_collections(client, collection_name)

        # Create indexes and load collection
        index_times = self._setup_indexes_and_load(client, collection_name)

        # Add collection release info to index times
        index_times.update(release_times)

        # Verify collection and indexes are ready
        self._verify_collection_ready(client, collection_name)

        # Warmup phase
        self._run_warmup(client, collection_name, queries_df)

        # Run concurrency sweep
        if len(self.concurrency_levels) > 1 or self.auto_extend_concurrency:
            results_df = self._run_concurrency_sweep(client, collection_name, queries_df, index_times)
        else:
            # Single concurrency run
            self.concurrency = self.concurrency_levels[0]
            results_df = self._run_benchmark(client, collection_name, queries_df)
            # Add index creation times to results
            for col, time_val in index_times.items():
                results_df[col] = time_val

        # Save results
        save_parquet(results_df, output_file)

        # Log summary statistics
        if len(self.concurrency_levels) > 1 or self.auto_extend_concurrency:
            self._log_concurrency_sweep_summary(results_df)
        else:
            self._log_benchmark_summary(results_df)

        return results_df

    def _setup_indexes_and_load(self, client: MilvusGeoClient, collection_name: str) -> dict[str, float]:
        """Setup indexes and load collection, return timing information."""
        index_times = {}

        # Handle vector index
        if self.create_vector_index:
            # Check if vector index already exists and create if needed
            if client._has_vector_index(collection_name):
                logging.info("Vector index already exists, skipping creation")
                index_times['vector_index_time_s'] = 0.0
            else:
                logging.info("Creating vector index...")
                vector_time = client.create_vector_index(collection_name)
                index_times['vector_index_time_s'] = vector_time
        else:
            # Drop vector index if it exists and we don't want it
            if client._has_vector_index(collection_name):
                logging.info("Dropping existing vector index (--no-create-vector-index specified)...")
                drop_time = client.drop_vector_index(collection_name)
                index_times['vector_index_time_s'] = -drop_time  # Negative indicates drop
            else:
                logging.info("No vector index to drop, skipping vector index creation")
                index_times['vector_index_time_s'] = 0.0

        # Handle geo index
        if self.create_geo_index:
            # Check if geo index already exists and create if needed
            if client._has_geo_index(collection_name):
                logging.info("Geo index already exists, skipping creation")
                index_times['geo_index_time_s'] = 0.0
            else:
                logging.info("Creating geo index...")
                geo_time = client.create_geo_index(collection_name)
                index_times['geo_index_time_s'] = geo_time
        else:
            # Drop geo index if it exists and we don't want it
            if client._has_geo_index(collection_name):
                logging.info("Dropping existing geo index (--no-create-geo-index specified)...")
                drop_time = client.drop_geo_index(collection_name)
                index_times['geo_index_time_s'] = -drop_time  # Negative indicates drop
            else:
                logging.info("No geo index to drop, skipping geo index creation")
                index_times['geo_index_time_s'] = 0.0

        # Load collection to memory (always needed, either first time or after index drops)
        logging.info("Loading collection to memory...")
        load_time = client.load_collection(collection_name)
        index_times['collection_load_time_s'] = load_time

        # Wait for indexes to be ready if any were created
        if (self.create_vector_index and index_times['vector_index_time_s'] > 0) or \
           (self.create_geo_index and index_times['geo_index_time_s'] > 0):
            logging.info("Waiting for indexes to be ready...")
            fields_to_check = []
            if self.create_vector_index and index_times['vector_index_time_s'] > 0:
                fields_to_check.append("embedding")
            if self.create_geo_index and index_times['geo_index_time_s'] > 0:
                fields_to_check.append("location")

            # Wait for the newly created indexes
            if fields_to_check:
                self._wait_for_specific_indexes(client, collection_name, fields_to_check, timeout=300)

        return index_times

    def _wait_for_specific_indexes(self, client: MilvusGeoClient, collection_name: str, field_names: list[str], timeout: int = 300) -> bool:
        """Wait for specific indexes to be ready."""
        max_wait_time = timeout
        check_interval = 5
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            all_ready = True
            status_messages = []

            for field_name in field_names:
                is_ready = client.is_index_ready(collection_name, field_name)
                index_info = client.get_index_status(collection_name, field_name)

                if index_info:
                    total_rows = index_info.get("total_rows", 0)
                    indexed_rows = index_info.get("indexed_rows", 0)
                    progress_pct = (indexed_rows / total_rows * 100) if total_rows > 0 else 100
                    status = f"{field_name}: {progress_pct:.1f}% ({indexed_rows}/{total_rows})"
                else:
                    status = f"{field_name}: No index info"

                status_messages.append(status)

                if not is_ready:
                    all_ready = False

            logging.info(f"Index build progress - {', '.join(status_messages)}")

            if all_ready:
                logging.info(f"All specified indexes are ready for collection '{collection_name}'")
                return True

            time.sleep(check_interval)

        # Timeout reached
        elapsed = time.time() - start_time
        logging.warning(f"Timeout waiting for indexes after {elapsed:.1f}s, proceeding anyway")
        return False

    def _release_other_collections(self, client: MilvusGeoClient, target_collection: str) -> dict[str, float]:
        """Release all collections except the target collection."""
        logging.info("\n=== Releasing Other Collections ===")
        logging.info(f"Target collection for benchmark: '{target_collection}'")

        try:
            # Get currently loaded collections
            loaded_collections = client.get_loaded_collections()
            logging.info(f"Currently loaded collections: {loaded_collections}")

            # Release other collections
            release_times = client.release_other_collections(target_collection)

            # Prepare summary for logging
            release_summary = {}
            for collection, release_time in release_times.items():
                if release_time > 0:
                    release_summary[f"{collection}_release_time_s"] = release_time
                elif release_time == 0:
                    release_summary[f"{collection}_release_time_s"] = 0.0  # Not loaded
                else:
                    release_summary[f"{collection}_release_time_s"] = 0.0  # Failed

            if release_summary:
                total_released = len([t for t in release_times.values() if t > 0])
                total_time = sum(t for t in release_times.values() if t > 0)
                if total_released > 0:
                    logging.info(f"Memory cleanup completed: released {total_released} collections in {total_time:.2f}s")
                else:
                    logging.debug("No collections were released")
            else:
                logging.debug("No other collections needed to be released")

            return release_summary

        except Exception as e:
            logging.warning(f"Failed to release other collections: {e}")
            return {"other_collections_release_error": True}

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

            # Check index status for enabled fields only
            fields_to_check = []
            if self.create_vector_index:
                fields_to_check.append("embedding")
            if self.create_geo_index:
                fields_to_check.append("location")

            if not fields_to_check:
                logging.info("No indexes enabled, skipping index status check")
            else:
                logging.info(f"Checking status for enabled indexes: {fields_to_check}")
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
                        logging.warning(f"No index found for field '{field_name}' (but it should exist)")

            # Final readiness check with brief wait (only for enabled indexes)
            if fields_to_check:
                logging.info(f"Performing final index readiness check for: {fields_to_check}")
                if self._wait_for_specific_indexes(client, collection_name, fields_to_check, timeout=30):
                    logging.info("Collection verification completed - all systems ready")
                else:
                    logging.warning("Some indexes may still be building, but proceeding with benchmark")
            else:
                logging.info("No indexes to wait for, collection is ready for benchmarking")

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

    def _run_concurrency_sweep(
        self, client: MilvusGeoClient, collection_name: str, queries_df: pd.DataFrame, index_times: dict[str, float]
    ) -> pd.DataFrame:
        """Run benchmark across multiple concurrency levels."""
        logging.info(f"Starting concurrency sweep with levels: {self.concurrency_levels}")

        all_results = []
        concurrency_results = []

        # Extend concurrency levels if auto-scaling is enabled
        test_levels = self.concurrency_levels.copy()

        for i, level in enumerate(test_levels):
            logging.info(f"\n=== Testing Concurrency Level: {level} ===")

            self.concurrency = level
            results_df = self._run_benchmark(client, collection_name, queries_df)

            # Add concurrency level and index times to results
            results_df['concurrency_level'] = level
            for col, time_val in index_times.items():
                results_df[col] = time_val

            # Calculate metrics for this concurrency level
            metrics = self._calculate_concurrency_metrics(results_df, level)
            concurrency_results.append(metrics)

            logging.info(f"Concurrency {level}: QPS={metrics['qps']:.1f}, P99={metrics['p99_latency']:.1f}ms, Errors={metrics['error_rate']:.1%}")

            all_results.append(results_df)

            # Check if we should continue scaling
            if self.auto_extend_concurrency and i == len(test_levels) - 1:
                if self._should_continue_scaling(concurrency_results):
                    next_levels = self._generate_next_concurrency_levels(level, concurrency_results)
                    test_levels.extend(next_levels)
                    logging.info(f"Auto-extending concurrency levels: {next_levels}")

        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=True)

        # Find and log optimal concurrency
        optimal = self._find_optimal_concurrency(concurrency_results)
        logging.info(f"\n=== Optimal Concurrency: {optimal['concurrency']} (QPS: {optimal['qps']:.1f}) ===")

        return combined_results

    def _calculate_concurrency_metrics(self, results_df: pd.DataFrame, concurrency_level: int) -> dict[str, Any]:
        """Calculate metrics for a specific concurrency level."""
        successful_results = results_df[results_df["success"]]

        if len(successful_results) == 0:
            return {
                'concurrency': concurrency_level,
                'qps': 0.0,
                'p50_latency': 0.0,
                'p95_latency': 0.0,
                'p99_latency': 0.0,
                'error_rate': 1.0,
                'success_count': 0,
                'total_count': len(results_df)
            }

        query_times = successful_results["query_time_ms"].values

        # Calculate QPS
        if concurrency_level > 1 and "total_execution_time_s" in successful_results.columns:
            total_execution_time = successful_results["total_execution_time_s"].iloc[0]
            qps = len(successful_results) / total_execution_time if total_execution_time > 0 else 0
        else:
            total_time = query_times.sum() / 1000  # Convert to seconds
            qps = len(successful_results) / total_time if total_time > 0 else 0

        return {
            'concurrency': concurrency_level,
            'qps': qps,
            'p50_latency': pd.Series(query_times).quantile(0.50),
            'p95_latency': pd.Series(query_times).quantile(0.95),
            'p99_latency': pd.Series(query_times).quantile(0.99),
            'error_rate': (len(results_df) - len(successful_results)) / len(results_df),
            'success_count': len(successful_results),
            'total_count': len(results_df)
        }

    def _should_continue_scaling(self, concurrency_results: list[dict[str, Any]]) -> bool:
        """Determine if concurrency scaling should continue."""
        if len(concurrency_results) < 2:
            return True

        last_result = concurrency_results[-1]
        prev_result = concurrency_results[-2]

        # Check if we've hit the maximum limit
        if last_result['concurrency'] >= self.max_concurrency_limit:
            logging.info(f"Reached maximum concurrency limit: {self.max_concurrency_limit}")
            return False

        # Check QPS growth
        if prev_result['qps'] > 0:
            qps_growth = (last_result['qps'] - prev_result['qps']) / prev_result['qps'] * 100
        else:
            qps_growth = 100 if last_result['qps'] > 0 else 0

        # Check error rate
        if last_result['error_rate'] > self.max_error_rate:
            logging.info(f"Error rate too high: {last_result['error_rate']:.1%} > {self.max_error_rate:.1%}")
            return False

        # Check latency (P99 should be less than 80% of timeout)
        if last_result['p99_latency'] > self.timeout * 800:  # 80% of timeout in ms
            logging.info(f"P99 latency too high: {last_result['p99_latency']:.1f}ms")
            return False

        # Check QPS growth
        if qps_growth < self.qps_growth_threshold:
            logging.info(f"QPS growth below threshold: {qps_growth:.1f}% < {self.qps_growth_threshold}%")
            return False

        logging.info(f"Continuing scaling - QPS growth: {qps_growth:.1f}%")
        return True

    def _generate_next_concurrency_levels(self, current_max: int, concurrency_results: list[dict[str, Any]]) -> list[int]:
        """Generate next concurrency levels for testing."""
        # Simple doubling strategy, but could be made more sophisticated
        next_level = current_max * 2

        # Don't exceed the maximum limit
        if next_level > self.max_concurrency_limit:
            if current_max < self.max_concurrency_limit:
                return [self.max_concurrency_limit]
            else:
                return []

        return [next_level]

    def _find_optimal_concurrency(self, concurrency_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Find the concurrency level with maximum QPS."""
        if not concurrency_results:
            return {'concurrency': 1, 'qps': 0.0}

        # Find the result with maximum QPS
        optimal = max(concurrency_results, key=lambda x: x['qps'])
        return optimal

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

    def _log_concurrency_sweep_summary(self, results_df: pd.DataFrame) -> None:
        """Log summary statistics for concurrency sweep."""
        logging.info("\n=== Concurrency Sweep Results ===")
        logging.info("━" * 80)
        logging.info(f"{'Concurrency':<11} | {'QPS':<8} | {'P50(ms)':<7} | {'P99(ms)':<7} | {'Errors':<6}")
        logging.info("─" * 80)

        # Group by concurrency level
        concurrency_levels = sorted(results_df['concurrency_level'].unique())
        optimal_qps = 0
        optimal_concurrency = 0

        for level in concurrency_levels:
            level_data = results_df[results_df['concurrency_level'] == level]
            metrics = self._calculate_concurrency_metrics(level_data, level)

            if metrics['qps'] > optimal_qps:
                optimal_qps = metrics['qps']
                optimal_concurrency = level

            marker = " ← QPS declining" if level > optimal_concurrency and metrics['qps'] < optimal_qps * 0.95 else ""
            auto_marker = " (auto)" if level not in self.concurrency_levels else ""

            logging.info(
                f"{level}{auto_marker:<11} | {metrics['qps']:<8.1f} | {metrics['p50_latency']:<7.1f} | "
                f"{metrics['p99_latency']:<7.1f} | {metrics['error_rate']:<6.1%}{marker}"
            )

        logging.info("\n" + "━" * 80)
        logging.info(f"Optimal Concurrency: {optimal_concurrency} (Maximum QPS: {optimal_qps:.1f})")

        # Log index operation times if available
        if 'vector_index_time_s' in results_df.columns:
            vector_time = results_df['vector_index_time_s'].iloc[0]
            geo_time = results_df['geo_index_time_s'].iloc[0]
            load_time = results_df['collection_load_time_s'].iloc[0]

            logging.info("\n=== Index Operation Times ===")

            # Handle vector index operation
            if vector_time > 0:
                logging.info(f"Vector index creation: {vector_time:.2f}s")
            elif vector_time < 0:
                logging.info(f"Vector index drop: {abs(vector_time):.2f}s")
            else:
                logging.info("Vector index: no operation (already exists or not requested)")

            # Handle geo index operation
            if geo_time > 0:
                logging.info(f"Geo index creation: {geo_time:.2f}s")
            elif geo_time < 0:
                logging.info(f"Geo index drop: {abs(geo_time):.2f}s")
            else:
                logging.info("Geo index: no operation (already exists or not requested)")

            logging.info(f"Collection load: {load_time:.2f}s")

            # Log other collection releases at debug level only
            release_keys = [k for k in results_df.columns if k.endswith('_release_time_s') and not k.startswith('collection_')]
            if release_keys:
                logging.debug("\n=== Other Collections Released ===")
                for key in release_keys:
                    collection_name = key.replace('_release_time_s', '')
                    release_time = results_df[key].iloc[0]
                    if release_time > 0:
                        logging.debug(f"{collection_name}: {release_time:.2f}s")
                    elif release_time == 0:
                        logging.debug(f"{collection_name}: not loaded (skipped)")

            total_time = abs(vector_time) + abs(geo_time) + load_time
            logging.info(f"\nTotal setup time: {total_time:.2f}s")

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
        concurrency = getattr(self, 'concurrency', 1)
        if concurrency > 1 and "total_execution_time_s" in successful_results.columns:
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

        # Log index operation times if available
        if 'vector_index_time_s' in results_df.columns:
            vector_time = results_df['vector_index_time_s'].iloc[0]
            geo_time = results_df['geo_index_time_s'].iloc[0]
            load_time = results_df['collection_load_time_s'].iloc[0]

            logging.info("\n=== Index Operation Times ===")

            # Handle vector index operation
            if vector_time > 0:
                logging.info(f"Vector index creation: {vector_time:.2f}s")
            elif vector_time < 0:
                logging.info(f"Vector index drop: {abs(vector_time):.2f}s")
            else:
                logging.info("Vector index: no operation (already exists or not requested)")

            # Handle geo index operation
            if geo_time > 0:
                logging.info(f"Geo index creation: {geo_time:.2f}s")
            elif geo_time < 0:
                logging.info(f"Geo index drop: {abs(geo_time):.2f}s")
            else:
                logging.info("Geo index: no operation (already exists or not requested)")

            logging.info(f"Collection load: {load_time:.2f}s")

            # Log other collection releases at debug level only
            release_keys = [k for k in results_df.columns if k.endswith('_release_time_s') and not k.startswith('collection_')]
            if release_keys:
                logging.debug("\n=== Other Collections Released ===")
                for key in release_keys:
                    collection_name = key.replace('_release_time_s', '')
                    release_time = results_df[key].iloc[0]
                    if release_time > 0:
                        logging.debug(f"{collection_name}: {release_time:.2f}s")
                    elif release_time == 0:
                        logging.debug(f"{collection_name}: not loaded (skipped)")

            total_time = abs(vector_time) + abs(geo_time) + load_time
            logging.info(f"\nTotal setup time: {total_time:.2f}s")

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
