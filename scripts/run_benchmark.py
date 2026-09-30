import argparse
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.evaluation.benchmark import BenchmarkRunner
from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger("nexus.scripts.benchmark")


def main():
    parser = argparse.ArgumentParser(description="Run complete Nexus Retrieval benchmark evaluation.")
    parser.add_argument(
        "--dataset",
        choices=["coir", "samsung_demo"],
        default="coir",
        help="Dataset split to evaluate",
    )
    parser.add_argument("--split", type=str, default="test", help="Split name")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries for fast sanity runs")
    parser.add_argument("--force", action="store_true", help="Force re-encode embeddings")
    args = parser.parse_args()

    logger.info(f"Starting benchmark evaluation (Dataset: {args.dataset}, Limit: {args.limit})...")
    runner = BenchmarkRunner(dataset_name=args.dataset, split=args.split, limit=args.limit)
    report = runner.run_all(force_rebuild=args.force)

    settings = get_settings()
    json_path = settings.results_dir / "benchmark.json"
    md_path = settings.results_dir / "benchmark.md"

    runner.save_reports(report, json_path, md_path)
    logger.info("Benchmark complete!")


if __name__ == "__main__":
    main()
