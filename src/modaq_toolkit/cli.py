"""Command line interface for MODAQ toolkit."""

import argparse
import sys
from pathlib import Path

from .parser import process_mcap_files


def main(args: list[str] | None = None) -> int:
    """Extract and Transform Raw MODAQ data into a standardized parquet files"""
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Extract and Transform Raw MODAQ data into a standardized parquet files"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing MCAP files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("./data/"),
        help="Directory for output (default: ./data)",
    )
    parser.add_argument(
        "--async",
        dest="async_processing",
        action="store_true",
        default=False,
        help="Enable asynchronous processing of MCAP files (default: False)",
    )
    parser.add_argument(
        "--skip-topics",
        dest="topics_to_skip",
        type=str,
        nargs="*",
        default=None,
        help="List of topic names to skip during processing (space-separated)",
    )
    parser.add_argument(
        "--stage1",
        dest="process_stage1",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Process stage 1 output (one-to-one ROS message structure). Default: False",
    )
    parser.add_argument(
        "--stage2",
        dest="process_stage2",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Process stage 2 output (unpacked arrays for analysis). Default: True",
    )
    parser.add_argument(
        "--stage1-dir",
        dest="stage1_dir",
        type=str,
        default="a1_one_to_one",
        help="Directory name for stage 1 output (default: a1_one_to_one)",
    )
    parser.add_argument(
        "--stage2-dir",
        dest="stage2_dir",
        type=str,
        default="a2_unpacked",
        help="Directory name for stage 2 output (default: a2_unpacked)",
    )

    parsed_args = parser.parse_args(args)
    process_mcap_files(
        parsed_args.input_dir,
        parsed_args.output_dir,
        async_processing=parsed_args.async_processing,
        topics_to_skip=parsed_args.topics_to_skip,
        process_stage1=parsed_args.process_stage1,
        process_stage2=parsed_args.process_stage2,
        stage1_dir=parsed_args.stage1_dir,
        stage2_dir=parsed_args.stage2_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
