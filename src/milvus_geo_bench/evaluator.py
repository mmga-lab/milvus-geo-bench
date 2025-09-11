"""
Evaluation module for benchmark results.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import load_parquet


class Evaluator:
    """Evaluate benchmark results against ground truth."""

    def __init__(self):
        pass

    def evaluate_results(self, results_file: str, ground_truth_file: str) -> dict[str, Any]:
        """Evaluate benchmark results against ground truth."""

        # Load data
        results_df = load_parquet(results_file)
        ground_truth_df = load_parquet(ground_truth_file)

        logging.info(f"Loaded {len(results_df)} benchmark results")
        logging.info(f"Loaded {len(ground_truth_df)} ground truth records")

        # Filter successful queries only
        successful_results = results_df[results_df["success"]].copy()

        if len(successful_results) == 0:
            logging.error("No successful queries to evaluate")
            return {
                "error": "No successful queries found",
                "total_queries": len(results_df),
                "successful_queries": 0,
            }

        # Merge with ground truth
        merged_df = successful_results.merge(
            ground_truth_df, on="query_id", how="inner", suffixes=("_result", "_truth")
        )

        if len(merged_df) == 0:
            logging.error("No matching queries found between results and ground truth")
            return {
                "error": "No matching queries found",
                "total_queries": len(results_df),
                "successful_queries": len(successful_results),
            }

        logging.info(f"Evaluating {len(merged_df)} queries")

        # Calculate metrics for each query
        query_metrics = []
        total_queries_with_results = 0
        total_queries_no_results = 0

        for _, row in merged_df.iterrows():
            predicted_ids = row["result_ids_result"]
            true_ids = row["result_ids_truth"]

            # Check if query returned any results
            has_results = predicted_ids is not None and len(predicted_ids) > 0
            if has_results:
                total_queries_with_results += 1
            else:
                total_queries_no_results += 1

            metrics = self._calculate_query_metrics(
                predicted_ids=predicted_ids,
                true_ids=true_ids,
                query_id=row["query_id"],
            )
            query_metrics.append(metrics)

        logging.info(f"Queries with results: {total_queries_with_results}")
        logging.info(f"Queries with no results: {total_queries_no_results}")

        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(query_metrics, merged_df)

        # Add performance metrics
        performance_metrics = self._calculate_performance_metrics(successful_results)
        overall_metrics.update(performance_metrics)

        # Add metadata
        overall_metrics.update(
            {
                "evaluation_timestamp": datetime.now().isoformat(),
                "total_queries": len(results_df),
                "successful_queries": len(successful_results),
                "evaluated_queries": len(merged_df),
                "failed_queries": len(results_df) - len(successful_results),
            }
        )

        return overall_metrics

    def _calculate_query_metrics(
        self, predicted_ids: list[int], true_ids: list[int], query_id: int
    ) -> dict[str, Any]:
        """Calculate precision, recall, and F1 for a single query."""

        # Handle potential None or empty values safely
        predicted_set = (
            set(predicted_ids) if predicted_ids is not None and len(predicted_ids) > 0 else set()
        )
        true_set = set(true_ids) if true_ids is not None and len(true_ids) > 0 else set()

        # Calculate intersection
        intersection = predicted_set.intersection(true_set)

        # Calculate metrics
        precision = len(intersection) / len(predicted_set) if len(predicted_set) > 0 else 0.0
        recall = len(intersection) / len(true_set) if len(true_set) > 0 else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        )

        fp_count = len(predicted_set - true_set)

        # Log queries with false positives for debugging
        if fp_count > 0:
            logging.debug(
                f"Query {query_id}: {fp_count} false positives (precision: {precision:.3f})"
            )

        return {
            "query_id": query_id,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": len(intersection),
            "false_positives": fp_count,
            "false_negatives": len(true_set - predicted_set),
            "predicted_count": len(predicted_set),
            "true_count": len(true_set),
        }

    def _calculate_overall_metrics(
        self, query_metrics: list[dict[str, Any]], merged_df: pd.DataFrame
    ) -> dict[str, Any]:
        """Calculate overall evaluation metrics."""

        if not query_metrics:
            return {"error": "No query metrics available"}

        # Extract metric arrays
        precisions = [m["precision"] for m in query_metrics]
        recalls = [m["recall"] for m in query_metrics]
        f1_scores = [m["f1_score"] for m in query_metrics]

        # Calculate macro averages
        macro_precision = np.mean(precisions)
        macro_recall = np.mean(recalls)
        macro_f1 = np.mean(f1_scores)

        # Calculate micro averages
        total_tp = sum(m["true_positives"] for m in query_metrics)
        total_fp = sum(m["false_positives"] for m in query_metrics)
        total_fn = sum(m["false_negatives"] for m in query_metrics)

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (
            2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0.0
        )

        # Calculate distribution statistics
        precision_stats = self._calculate_distribution_stats(precisions)
        recall_stats = self._calculate_distribution_stats(recalls)
        f1_stats = self._calculate_distribution_stats(f1_scores)

        # Identify queries with false positives (wrong results)
        queries_with_errors = []
        fp_count = 0
        for metric in query_metrics:
            if metric["false_positives"] > 0:
                fp_count += 1
                queries_with_errors.append(
                    {
                        "query_id": metric["query_id"],
                        "precision": metric["precision"],
                        "false_positives": metric["false_positives"],
                        "predicted_count": metric["predicted_count"],
                        "true_count": metric["true_count"],
                    }
                )

        # Log debug information
        logging.info(
            f"Found {fp_count} queries with false positives out of {len(query_metrics)} total queries"
        )
        logging.info(f"Total false positives: {total_fp}")

        # Sort by false positives count (worst first)
        queries_with_errors.sort(key=lambda x: x["false_positives"], reverse=True)

        return {
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "precision_stats": precision_stats,
            "recall_stats": recall_stats,
            "f1_stats": f1_stats,
            "total_true_positives": total_tp,
            "total_false_positives": total_fp,
            "total_false_negatives": total_fn,
            "queries_with_errors": queries_with_errors,
            "error_query_count": len(queries_with_errors),
            "query_metrics": query_metrics,  # Include individual query metrics for analysis
        }

    def _calculate_throughput(self, results_df: pd.DataFrame, query_times: np.ndarray) -> float:
        """Calculate throughput considering concurrent vs serial execution."""
        if "total_execution_time_s" in results_df.columns:
            # Concurrent execution: use actual total execution time
            total_time_s = results_df["total_execution_time_s"].iloc[0]
            return len(results_df) / total_time_s if total_time_s > 0 else 0.0
        else:
            # Serial execution: use sum of query times
            total_time_s = np.sum(query_times) / 1000
            return len(results_df) / total_time_s if total_time_s > 0 else 0.0

    def _calculate_distribution_stats(self, values: list[float]) -> dict[str, float]:
        """Calculate distribution statistics for a list of values."""
        if not values:
            return {}

        values_array = np.array(values)
        return {
            "mean": float(np.mean(values_array)),
            "median": float(np.median(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "q25": float(np.percentile(values_array, 25)),
            "q75": float(np.percentile(values_array, 75)),
            "q95": float(np.percentile(values_array, 95)),
            "q99": float(np.percentile(values_array, 99)),
        }

    def _calculate_performance_metrics(self, results_df: pd.DataFrame) -> dict[str, Any]:
        """Calculate performance metrics from benchmark results."""

        query_times = results_df["query_time_ms"].values
        result_counts = results_df["result_count"].values

        performance_stats = {
            "query_time_stats": self._calculate_distribution_stats(query_times.tolist()),
            "result_count_stats": self._calculate_distribution_stats(result_counts.tolist()),
            "total_execution_time_ms": float(np.sum(query_times)),
            "average_throughput_qps": self._calculate_throughput(results_df, query_times),
        }

        return performance_stats

    def generate_report(
        self,
        metrics: dict[str, Any],
        output_format: str = "markdown",
        output_file: str | None = None,
    ) -> str:
        """Generate evaluation report."""

        if "error" in metrics:
            report_content = f"# Evaluation Report\n\n**Error:** {metrics['error']}\n"
        elif output_format.lower() == "markdown":
            report_content = self._generate_markdown_report(metrics)
        elif output_format.lower() == "json":
            report_content = json.dumps(metrics, indent=2)
        else:
            report_content = str(metrics)

        # Save to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                f.write(report_content)
            logging.info(f"Report saved to {output_path}")

        return report_content

    def _generate_markdown_report(self, metrics: dict[str, Any]) -> str:
        """Generate markdown format report."""

        report = []
        report.append("# Milvus Geo Search Benchmark Evaluation Report")
        report.append(f"\n**Generated:** {metrics.get('evaluation_timestamp', 'N/A')}")
        report.append("")

        # Summary
        report.append("## Summary")
        report.append(f"- Total queries: {metrics.get('total_queries', 'N/A')}")
        report.append(f"- Successful queries: {metrics.get('successful_queries', 'N/A')}")
        report.append(f"- Evaluated queries: {metrics.get('evaluated_queries', 'N/A')}")
        report.append(f"- Failed queries: {metrics.get('failed_queries', 'N/A')}")

        if metrics.get("successful_queries", 0) > 0:
            success_rate = (
                metrics.get("successful_queries", 0) / metrics.get("total_queries", 1)
            ) * 100
            report.append(f"- Success rate: {success_rate:.2f}%")
        report.append("")

        # Accuracy Metrics
        report.append("## Accuracy Metrics")
        report.append("")
        report.append("### Overall Performance")
        report.append(f"- **Macro Precision:** {metrics.get('macro_precision', 0):.4f}")
        report.append(f"- **Macro Recall:** {metrics.get('macro_recall', 0):.4f}")
        report.append(f"- **Macro F1-Score:** {metrics.get('macro_f1', 0):.4f}")
        report.append(f"- **Micro Precision:** {metrics.get('micro_precision', 0):.4f}")
        report.append(f"- **Micro Recall:** {metrics.get('micro_recall', 0):.4f}")
        report.append(f"- **Micro F1-Score:** {metrics.get('micro_f1', 0):.4f}")
        report.append("")

        # Precision Statistics
        if "precision_stats" in metrics:
            report.append("### Precision Distribution")
            self._add_stats_table(report, metrics["precision_stats"])

        # Recall Statistics
        if "recall_stats" in metrics:
            report.append("### Recall Distribution")
            self._add_stats_table(report, metrics["recall_stats"])

        # F1 Statistics
        if "f1_stats" in metrics:
            report.append("### F1-Score Distribution")
            self._add_stats_table(report, metrics["f1_stats"])

        # Performance Metrics
        report.append("## Performance Metrics")
        report.append("")

        if "query_time_stats" in metrics:
            report.append("### Query Time Statistics (ms)")
            self._add_stats_table(report, metrics["query_time_stats"])

        if "result_count_stats" in metrics:
            report.append("### Result Count Statistics")
            self._add_stats_table(report, metrics["result_count_stats"])

        # Throughput
        throughput = metrics.get("average_throughput_qps", 0)
        total_time = metrics.get("total_execution_time_ms", 0) / 1000
        report.append(f"- **Average Throughput:** {throughput:.2f} queries/second")
        report.append(f"- **Total Execution Time:** {total_time:.2f} seconds")
        report.append("")

        # Confusion Matrix Summary
        report.append("## Confusion Matrix Summary")
        report.append(f"- True Positives: {metrics.get('total_true_positives', 0)}")
        report.append(f"- False Positives: {metrics.get('total_false_positives', 0)}")
        report.append(f"- False Negatives: {metrics.get('total_false_negatives', 0)}")
        report.append("")

        # Queries with Errors (False Positives)
        error_queries = metrics.get("queries_with_errors", [])
        if error_queries:
            report.append("## Queries with Wrong Results (False Positives)")
            report.append(f"Found {len(error_queries)} queries that returned incorrect results:")
            report.append("")
            report.append("| Query ID | Precision | False Positives | Predicted | Expected |")
            report.append("|----------|-----------|-----------------|-----------|----------|")

            # Show top 20 worst queries
            for query in error_queries[:20]:
                report.append(
                    f"| {query['query_id']} | "
                    f"{query['precision']:.4f} | "
                    f"{query['false_positives']} | "
                    f"{query['predicted_count']} | "
                    f"{query['true_count']} |"
                )

            if len(error_queries) > 20:
                report.append("| ... | ... | ... | ... | ... |")
                report.append(f"*(Showing top 20 of {len(error_queries)} queries with errors)*")
            report.append("")

            # Summary statistics for error queries
            avg_fp = sum(q["false_positives"] for q in error_queries) / len(error_queries)
            max_fp = max(q["false_positives"] for q in error_queries)
            report.append("### Error Query Statistics")
            report.append(f"- Average false positives per error query: {avg_fp:.1f}")
            report.append(f"- Maximum false positives in a single query: {max_fp}")
            report.append(
                f"- Percentage of queries with errors: {len(error_queries) / metrics.get('evaluated_queries', 1) * 100:.2f}%"
            )
            report.append("")

        return "\n".join(report)

    def _add_stats_table(self, report: list[str], stats: dict[str, float]) -> None:
        """Add statistics table to report."""
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| Mean | {stats.get('mean', 0):.4f} |")
        report.append(f"| Median | {stats.get('median', 0):.4f} |")
        report.append(f"| Std Dev | {stats.get('std', 0):.4f} |")
        report.append(f"| Min | {stats.get('min', 0):.4f} |")
        report.append(f"| Max | {stats.get('max', 0):.4f} |")
        report.append(f"| Q25 | {stats.get('q25', 0):.4f} |")
        report.append(f"| Q75 | {stats.get('q75', 0):.4f} |")
        report.append(f"| P95 | {stats.get('q95', 0):.4f} |")
        report.append(f"| P99 | {stats.get('q99', 0):.4f} |")
        report.append("")

    def print_summary(self, metrics: dict[str, Any]) -> None:
        """Print evaluation summary to console."""

        if "error" in metrics:
            print(f"Evaluation Error: {metrics['error']}")
            return

        print("\n=== Benchmark Evaluation Summary ===")
        print(f"Total Queries: {metrics.get('total_queries', 'N/A')}")
        print(f"Successful Queries: {metrics.get('successful_queries', 'N/A')}")
        print(
            f"Success Rate: {(metrics.get('successful_queries', 0) / metrics.get('total_queries', 1) * 100):.2f}%"
        )
        print()

        print("Accuracy Metrics:")
        print(f"  Macro Precision: {metrics.get('macro_precision', 0):.4f}")
        print(f"  Macro Recall: {metrics.get('macro_recall', 0):.4f}")
        print(f"  Macro F1-Score: {metrics.get('macro_f1', 0):.4f}")
        print()

        # Show error queries information
        error_query_count = metrics.get("error_query_count", 0)
        total_evaluated = metrics.get("evaluated_queries", 0)
        total_fp = metrics.get("total_false_positives", 0)

        # Always show detailed precision analysis
        print("Precision Analysis:")
        print(f"  Total false positives across all queries: {total_fp}")
        print(f"  Queries with false positives: {error_query_count}")

        # Show precision distribution
        print("  Precision distribution:")
        print(
            f"    Perfect (1.0): {sum(1 for m in metrics.get('query_metrics', []) if m.get('precision', 0) == 1.0)} queries"
        )
        print(
            f"    Zero (0.0): {sum(1 for m in metrics.get('query_metrics', []) if m.get('precision', 0) == 0.0)} queries"
        )
        print(
            f"    Partial (0.0-1.0): {sum(1 for m in metrics.get('query_metrics', []) if 0 < m.get('precision', 0) < 1.0)} queries"
        )

        if error_query_count > 0:
            print("⚠️  Error Analysis:")
            print(f"  Queries with wrong results: {error_query_count}")
            print(f"  Error rate: {error_query_count / total_evaluated * 100:.2f}%")
            print(f"  Total false positives: {metrics.get('total_false_positives', 0)}")

            # Show top 5 worst queries
            error_queries = metrics.get("queries_with_errors", [])
            if error_queries:
                print("  Top 5 worst queries (by false positive count):")
                for _, query in enumerate(error_queries[:5]):
                    print(
                        f"    Query {query['query_id']}: {query['false_positives']} wrong results "
                        f"(precision: {query['precision']:.3f})"
                    )
            print()

        if "query_time_stats" in metrics:
            time_stats = metrics["query_time_stats"]
            print("Performance:")
            print(f"  Average Query Time: {time_stats.get('mean', 0):.2f}ms")
            print(f"  Median Query Time: {time_stats.get('median', 0):.2f}ms")
            print(f"  P95 Query Time: {time_stats.get('q95', 0):.2f}ms")
            print(f"  Throughput: {metrics.get('average_throughput_qps', 0):.2f} QPS")
        print()
