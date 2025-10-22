import asyncio
import itertools
import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from .message_processing import (
    MessageProcessor,
    expand_array_columns_vertically,
    is_array_column,
    parse_ros_message_definition,
)

logger = logging.getLogger(__name__)


@dataclass
class TopicTiming:
    """Stores timing information for a topic."""

    start_time: datetime
    end_time: datetime
    number_of_samples: int
    duration: float
    mean_rate: float
    max_rate: float
    min_rate: float
    rate_std: float

    def format_for_display(self) -> tuple:
        """Returns formatted strings for display in timing summary."""
        return (
            self.start_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
            self.end_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
            self.number_of_samples,
            self.duration,
            self.mean_rate,
            self.max_rate,
            self.min_rate,
            self.rate_std,
        )


@dataclass
class TopicMetadata:
    """Stores metadata for a topic."""

    topic: str
    source_file: str
    n_messages: int
    n_columns: int
    columns: list
    dtypes: dict
    schema: dict
    processing_stage: str
    parquet_file: str


class MCAPParser:
    """Parses MCAP files and converts them to parquet format with metadata or returns dataframes in memory."""

    # Class-level configuration
    TIME_WARNING_THRESHOLD_SECONDS: float = (
        1.0  # Threshold for timing misalignment warnings
    )

    def __init__(self, mcap_path: Path, topics_to_skip: list[str] | None = None):
        self.mcap_path = mcap_path
        self.processors: dict[str, MessageProcessor] = {}
        self.schemas_by_topic: dict[str, dict] = {}
        self.dataframes: dict[str, pd.DataFrame] = {}
        self.topics_to_skip = topics_to_skip if topics_to_skip else []

    def _process_channel(self, channel, schema, summary) -> None:
        """Process a single channel from the MCAP file."""
        self.schemas_by_topic[channel.topic] = parse_ros_message_definition(schema.data)
        self.processors[channel.topic] = MessageProcessor(
            self.schemas_by_topic[channel.topic]
        )

    def _process_message(self, channel, decoded) -> None:
        """Process a single message from a channel."""
        if decoded is None:
            logger.warning(f"Could not decode message for topic {channel.topic}")
            return

        try:
            self.processors[channel.topic].process_message(decoded)
        except Exception as e:
            raise RuntimeError(
                f"Failed to process message for topic {channel.topic}: {e}"
            )

    def read_mcap(self) -> None:
        """Read and process the MCAP file."""
        logger.info(f"Processing MCAP file: {self.mcap_path}")

        with open(self.mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            summary = reader.get_summary()

            if summary is None:
                raise ValueError("Could not read summary from MCAP file")

            # Process channels
            topic_names = [channel.topic for channel in summary.channels.values()]
            logger.info(f"Found {len(topic_names)} topics: {', '.join(topic_names)}")

            for channel in summary.channels.values():
                schema = summary.schemas[channel.schema_id]
                self._process_channel(channel, schema, summary)

            # Process messages
            for schema, channel, message, decoded in reader.iter_decoded_messages():
                self._process_message(channel, decoded)

            # Create dataframes
            for topic, processor in self.processors.items():
                if topic not in self.topics_to_skip:
                    self.dataframes[topic] = processor.get_dataframe()
                    logger.info(
                        f"Topic {topic} DataFrame shape: {self.dataframes[topic].shape}"
                    )

    def _process_dataframe_for_stage2(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, float]:
        """Process a dataframe for stage 2, returning the processed df and sample rate."""
        object_columns = [col for col in df.columns if is_array_column(df[col])]

        if object_columns:
            logger.debug("Found arrays, expanding for a2 processing")
            df = expand_array_columns_vertically(df)
            df["time"] = pd.to_datetime(
                df["system_time"], origin="unix", unit="ns", utc=True
            )
            df = df.set_index("time")
            df = df.drop(["sec", "nanosec", "frame_id", "timestamp"], axis="columns")

            time_diffs = df.index.to_series().diff().dt.total_seconds()
            return df, 1 / time_diffs.mean()
        else:
            df["time"] = pd.to_datetime(
                df["timestamp"], origin="unix", unit="s", utc=True
            )
            df = df.set_index("time")
            return df, None

    def _get_topic_timing(self, df: pd.DataFrame) -> TopicTiming:
        """Calculate timing information for a topic including sample rate statistics."""
        time_diffs = df.index.to_series().diff().dt.total_seconds()
        rates = 1 / time_diffs.dropna()  # Convert to Hz and remove NaN from first diff

        return TopicTiming(
            start_time=df.index.min(),
            end_time=df.index.max(),
            number_of_samples=len(df),
            duration=(df.index.max() - df.index.min()).total_seconds(),
            mean_rate=rates.mean(),
            max_rate=rates.max(),
            min_rate=rates.min(),
            rate_std=rates.std(),
        )

    def _check_timing_misalignments(self, time_tracker: dict[str, TopicTiming]) -> None:
        """Check for timing misalignments between topics."""
        if len(time_tracker) <= 1:
            return

        # Filter out zero-duration topics for timing calculations
        non_zero_topics = {
            topic: stats for topic, stats in time_tracker.items() if stats.duration > 0
        }

        excluded_topics = set(time_tracker.keys()) - set(non_zero_topics.keys())
        if excluded_topics:
            logger.info(
                f"Excluding zero-duration topics from timing alignment check: {', '.join(excluded_topics)}"
            )

        # Find earliest and latest start/end times and their corresponding topics
        start_times = {
            topic: stats.start_time for topic, stats in non_zero_topics.items()
        }
        end_times = {topic: stats.end_time for topic, stats in non_zero_topics.items()}

        earliest_start = min(start_times.values())
        latest_start = max(start_times.values())
        earliest_end = min(end_times.values())
        latest_end = max(end_times.values())

        earliest_start_topics = [
            topic for topic, time in start_times.items() if time == earliest_start
        ]
        latest_start_topics = [
            topic for topic, time in start_times.items() if time == latest_start
        ]
        earliest_end_topics = [
            topic for topic, time in end_times.items() if time == earliest_end
        ]
        latest_end_topics = [
            topic for topic, time in end_times.items() if time == latest_end
        ]

        max_start_diff = latest_start - earliest_start
        max_end_diff = latest_end - earliest_end

        if max_start_diff.total_seconds() > self.TIME_WARNING_THRESHOLD_SECONDS:
            logger.warning(
                f"Start time misalignment detected!\n"
                f"  Expected maximum difference: {self.TIME_WARNING_THRESHOLD_SECONDS:.3f} seconds\n"
                f"  Actual difference: {max_start_diff.total_seconds():.3f} seconds\n"
                f"  Earliest starting topics: {', '.join(earliest_start_topics)} at {earliest_start}\n"
                f"  Latest starting topics: {', '.join(latest_start_topics)} at {latest_start}"
            )

        if max_end_diff.total_seconds() > self.TIME_WARNING_THRESHOLD_SECONDS:
            logger.warning(
                f"End time misalignment detected!\n"
                f"  Expected maximum difference: {self.TIME_WARNING_THRESHOLD_SECONDS:.3f} seconds\n"
                f"  Actual difference: {max_end_diff.total_seconds():.3f} seconds\n"
                f"  Earliest ending topics: {', '.join(earliest_end_topics)} at {earliest_end}\n"
                f"  Latest ending topics: {', '.join(latest_end_topics)} at {latest_end}"
            )

    def get_dataframes(self, process_stage2: bool = False) -> dict[str, pd.DataFrame]:
        """
        Return a dictionary of processed dataframes without saving to disk.

        Args:
            process_stage2: If True, process dataframes for stage 2 (expand arrays, etc.)

        Returns:
            A dictionary with topic names as keys and processed dataframes as values
        """
        result = {}

        # Filter out empty dataframes
        non_empty_topics = {
            topic: df for topic, df in self.dataframes.items() if not df.empty
        }

        if not process_stage2:
            # Return the original dataframes
            return non_empty_topics

        # Process for stage 2 (expand arrays, add time index, etc.)
        for topic, df in non_empty_topics.items():
            processed_df, _ = self._process_dataframe_for_stage2(df.copy())
            result[topic] = processed_df

        return result

    def create_output(self, output_dir: Path, stage: str = "a1_one_to_one") -> None:
        """Create partitioned parquet files and metadata JSON for each topic."""
        stage_dir = output_dir / stage
        metadata_dir = output_dir / "metadata"
        stage_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running stage: {stage}")
        summary_metadata = {}
        time_tracker: dict[str, TopicTiming] = {}

        non_empty_topics = [
            topic for topic, df in self.dataframes.items() if not df.empty
        ]
        logger.info(f"Processing {len(non_empty_topics)} non-empty topics")

        for topic, df in self.dataframes.items():
            if df.empty:
                logger.debug(f"Skipping empty topic: {topic}")
                continue

            safe_topic = topic.replace("/", "_").lstrip("_")
            topic_dir = stage_dir / f"channel={safe_topic}"
            topic_dir.mkdir(exist_ok=True)

            if stage == "a2_real_data":
                df, sample_rate = self._process_dataframe_for_stage2(df)
                if sample_rate:
                    logger.debug(f"Sample rate for {topic}: {sample_rate:.3f} Hz")
                time_tracker[safe_topic] = self._get_topic_timing(df)

            # Save parquet file
            base_filename = f"{self.mcap_path.stem}.{safe_topic}"
            parquet_filename = f"{base_filename}.parquet"
            output_path = topic_dir / parquet_filename
            df.to_parquet(output_path)
            logger.debug(f"Saved {output_path}")

            if stage == "a1_one_to_one":
                self._save_topic_metadata(
                    topic, df, output_path, stage_dir, metadata_dir, summary_metadata
                )

        logger.info(f"Data saved to: {stage_dir}")

        if stage == "a2_real_data" and time_tracker:
            self._display_timing_summary(time_tracker)
            self._check_timing_misalignments(time_tracker)

        if stage == "a1_one_to_one":
            self._save_summary_metadata(summary_metadata, metadata_dir)

    def _save_topic_metadata(
        self,
        topic: str,
        df: pd.DataFrame,
        output_path: Path,
        stage_dir: Path,
        metadata_dir: Path,
        summary_metadata: dict,
    ) -> None:
        """Save metadata for a single topic."""
        topic_metadata = TopicMetadata(
            topic=topic,
            source_file=str(self.mcap_path),
            n_messages=len(df),
            n_columns=len(df.columns),
            columns=df.columns.tolist(),
            dtypes={col: str(df[col].dtype) for col in df.columns},
            schema=self.schemas_by_topic[topic],
            processing_stage="a1_one_to_one",
            parquet_file=str(output_path.relative_to(stage_dir)),
        )

        safe_topic = topic.replace("/", "_").lstrip("_")
        safe_topic_metadata_name = f"{self.mcap_path.stem}.{safe_topic}.metadata.json"
        topic_metadata_path = metadata_dir / safe_topic_metadata_name

        with open(topic_metadata_path, "w") as f:
            json.dump(vars(topic_metadata), f, indent=2)

        logger.debug(f"Saved metadata to {topic_metadata_path}")
        summary_metadata[topic] = vars(topic_metadata)

    def _save_summary_metadata(
        self, summary_metadata: dict, metadata_dir: Path
    ) -> None:
        """Save summary metadata for all topics."""
        summary = {
            "source_file": str(self.mcap_path),
            "n_topics": len(self.dataframes),
            "n_topics_with_data": sum(
                1 for df in self.dataframes.values() if not df.empty
            ),
            "creation_time": pd.Timestamp.now().isoformat(),
            "processing_stage": "a1_one_to_one",
            "topics": summary_metadata,
        }

        summary_path = metadata_dir / f"{self.mcap_path.stem}.summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.debug(f"Saved summary metadata to {summary_path}")

    def _display_timing_summary(self, time_tracker: dict[str, TopicTiming]) -> None:
        """Display timing summary table using tabulate for better formatting."""
        from tabulate import tabulate

        # Create DataFrame from timing data
        summary_data = []
        for topic, stats in time_tracker.items():
            formatted_stats = stats.format_for_display()
            summary_data.append(
                {
                    "Topic": topic,
                    "Start Time [UTC]": formatted_stats[0],
                    "End Time [UTC]": formatted_stats[1],
                    "Samples": formatted_stats[2],
                    "Duration (s)": formatted_stats[3],
                    "Mean Sample Rate [Hz]": formatted_stats[4],
                    "Max Sample Rate [Hz]": formatted_stats[5],
                    "Min Sample Rate [Hz]": formatted_stats[6],
                    "Sample Rate Std Dev [Hz]": formatted_stats[7],
                }
            )

        df_summary = pd.DataFrame(summary_data)

        # Format numeric columns
        df_summary["Duration (s)"] = df_summary["Duration (s)"].map("{:.2f}".format)
        df_summary["Mean Sample Rate [Hz]"] = df_summary["Mean Sample Rate [Hz]"].map(
            "{:.2f}".format
        )
        df_summary["Max Sample Rate [Hz]"] = df_summary["Max Sample Rate [Hz]"].map(
            "{:.2f}".format
        )
        df_summary["Min Sample Rate [Hz]"] = df_summary["Min Sample Rate [Hz]"].map(
            "{:.2f}".format
        )
        df_summary["Sample Rate Std Dev [Hz]"] = df_summary[
            "Sample Rate Std Dev [Hz]"
        ].map("{:.3f}".format)

        # Configure column alignments
        alignments = {
            "Topic": "left",
            "Start Time [UTC]": "left",
            "End Time [UTC]": "left",
            "Samples": "right",
            "Duration (s)": "right",
            "Mean Sample Rate [Hz]": "right",
            "Max Sample Rate [Hz]": "right",
            "Min Sample Rate [Hz]": "right",
            "Sample Rate Std Dev [Hz]": "right",
        }

        # Display the summary using tabulate
        print("\nTiming Summary:")
        print(
            tabulate(
                df_summary,
                headers="keys",
                tablefmt="pretty",
                showindex=False,
                colalign=[alignments[col] for col in df_summary.columns],
            )
        )


def parse_run_name_from_path(input_path: Path):
    final_dir = input_path.name
    return final_dir.replace("Bag_", "")


def get_mcap_files(input_path: Path) -> list[tuple[Path, str]]:
    """
    Recursively find all MCAP files in the input directory and its subdirectories.
    Returns a list of tuples containing (file_path, group_name).
    Group name is the parent folder name for nested files.
    """
    mcap_files = []

    for mcap_path in input_path.rglob("*.mcap"):
        # Get the relative path from input directory to the file
        rel_path = mcap_path.relative_to(input_path)

        # If file is in a subdirectory, use parent folder name as group
        # if len(rel_path.parts) > 1:
        #     group_name = rel_path.parts[0]
        # else:
        #     group_name = "default"
        group_name = parse_run_name_from_path(input_path)

        mcap_files.append((mcap_path, group_name))

    return sorted(mcap_files)


def process_mcap_and_get_dataframes(
    mcap_file: Path, process_stage2: bool = False
) -> dict[str, pd.DataFrame]:
    """
    Process a single MCAP file and return the dataframes without saving to disk.

    Args:
        mcap_file: Path to the MCAP file
        process_stage2: If True, process dataframes for stage 2 (expand arrays, etc.)

    Returns:
        A dictionary with topic names as keys and processed dataframes as values
    """

    logger.info(f"Processing file {mcap_file.name} for in-memory access")

    parser = MCAPParser(mcap_file)
    parser.read_mcap()

    # Get dataframes with or without stage 2 processing
    return parser.get_dataframes(process_stage2=process_stage2)


def process_mcap_dir_to_dataframes(
    input_dir: Path, process_stage2: bool = True, concat_by_topic: bool = True
) -> dict[str, dict[str, pd.DataFrame] | pd.DataFrame]:
    """
    Process all MCAP files in a directory and return them as a dictionary
    organized by group name.

    Args:
        input_dir: Path to directory containing MCAP files
        process_stage2: If True, process dataframes with stage 2 processing
        concat_by_topic: If True, concatenate all dataframes for each topic within a group
                        If False, return a dictionary of dataframes per topic

    Returns:
        A dictionary where:
        - Keys are group names
        - Values are either:
          - If concat_by_topic=True: Dictionary mapping topic names to concatenated dataframes
          - If concat_by_topic=False: Dictionary mapping topic names to lists of dataframes
    """
    # Find all MCAP files in directory
    mcap_files = get_mcap_files(input_dir)

    if not mcap_files:
        logger.warning(f"No MCAP files found in {input_dir}")
        return {}

    logger.info(f"Found {len(mcap_files)} MCAP files in {input_dir}")

    # Group files by their parent directory
    groups = {
        group: len(list(files))
        for group, files in itertools.groupby(mcap_files, key=lambda x: x[1])
    }

    for group, count in groups.items():
        logger.info(f"Group '{group}': {count} files")

    # Dictionary to store intermediate results
    # Structure: {group_name: {topic_name: [dataframe1, dataframe2, ...]}}
    temp_result = {}

    # Process each file
    for mcap_file, group_name in mcap_files:
        # Get dataframes from current file
        topic_dfs = process_mcap_and_get_dataframes(mcap_file, process_stage2)

        # Skip if no dataframes were found
        if not topic_dfs:
            continue

        # Add each topic's dataframe to the appropriate group
        for topic, df in topic_dfs.items():
            # Skip empty dataframes
            if df.empty:
                continue

            # Add source file information
            this_df = df.copy()
            # this_df["source_file"] = mcap_file.name

            # Create group entry if it doesn't exist
            if group_name not in temp_result:
                temp_result[group_name] = {}

            # Create topic entry if it doesn't exist
            if topic not in temp_result[group_name]:
                temp_result[group_name][topic] = []

            # Add dataframe to list
            temp_result[group_name][topic].append(this_df)

    # Process the temporary results based on concat_by_topic flag
    result = {}

    for group_name, topic_dfs_dict in temp_result.items():
        result[group_name] = {}

        if concat_by_topic:
            # Concatenate dataframes for each topic
            for topic, dfs in topic_dfs_dict.items():
                concat_df = pd.concat(dfs)
                concat_df = concat_df.sort_index()
                result[group_name][topic] = concat_df
        else:
            # Keep dataframes separate
            for topic, dfs in topic_dfs_dict.items():
                result[group_name][topic] = dfs

    return result


def process_single_file(mcap_file: Path, group: str, output_path: Path) -> None:
    """Process a single MCAP file - this function runs in its own process."""
    logger.info(f"\nProcessing file {mcap_file.name} from group '{group}'")

    # Create group-specific output directories
    group_output = output_path / group
    group_metadata = group_output / "metadata"
    (group_output / "a1_one_to_one").mkdir(parents=True, exist_ok=True)
    (group_output / "a2_real_data").mkdir(parents=True, exist_ok=True)
    group_metadata.mkdir(parents=True, exist_ok=True)

    parser = MCAPParser(mcap_file)
    parser.read_mcap()
    original_dataframes = parser.dataframes.copy()

    parser.create_output(group_output, stage="a1_one_to_one")
    logger.info("Processing expanded arrays")
    parser.dataframes = original_dataframes
    parser.create_output(group_output, stage="a2_real_data")

    logger.info(f"Completed processing {mcap_file.name}\n")
    return mcap_file.name


async def process_mcap_files_parallel(
    mcap_files: list[tuple[Path, str]],
    output_path: Path,
    max_workers: int | None = None,
) -> None:
    """Process MCAP files in parallel using ProcessPoolExecutor."""
    if max_workers is None:
        # Use CPU count - 1 to leave one core free for system tasks
        max_workers = max(1, multiprocessing.cpu_count() - 1)

    logger.info(f"Processing files using {max_workers} worker processes")

    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        # Create tasks for all files
        futures = [
            loop.run_in_executor(
                pool, process_single_file, mcap_file, group, output_path
            )
            for mcap_file, group in mcap_files
        ]

        # Process files and handle results as they complete
        for completed in asyncio.as_completed(futures):
            try:
                result = await completed
                if result:
                    logger.debug(f"Successfully processed: {result}")
            except Exception as e:
                logger.error(f"Process failed with error: {e!s}")


def process_mcap_files(
    input_dir: str, output_dir: str, async_processing: bool = False
) -> None:
    """
    Process all MCAP files in a directory and its subdirectories with single read and dual output.

    Args:
        input_dir: Input directory containing MCAP files
        output_dir: Output directory for processed files
        async_processing: If True, process files in parallel using multiple CPU cores
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Find all MCAP files in input directory and subdirectories
    mcap_files = get_mcap_files(input_path)
    logger.info(f"Found {len(mcap_files)} MCAP files in {input_path}")

    # Group files by their parent directory
    groups = {
        group: len(list(files))
        for group, files in itertools.groupby(mcap_files, key=lambda x: x[1])
    }

    for group, count in groups.items():
        logger.info(f"Group '{group}': {count} files")

    if async_processing:
        try:
            asyncio.run(process_mcap_files_parallel(mcap_files, output_path))
        except KeyboardInterrupt:
            logger.warning("\nProcessing interrupted by user")
            return
    else:
        # Sequential processing
        for mcap_file, group in mcap_files:
            try:
                process_single_file(mcap_file, group, output_path)
            except KeyboardInterrupt:
                logger.warning("\nProcessing interrupted by user")
                return
