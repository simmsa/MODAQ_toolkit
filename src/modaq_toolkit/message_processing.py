import logging
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_array_column(series):
    """Check if a pandas series contains numpy arrays"""
    if series.dtype == "object":
        first_val = series.dropna().iloc[0] if not series.isna().all() else None
        return isinstance(first_val, (np.ndarray, list))
    return False


def _normalize_to_array(value):
    """Normalize a value to an array, handling scalars and empty arrays.

    Returns None for values that should be skipped (empty arrays, None, NaN).
    Returns a list/array for valid values.

    This preserves preserves numpy scalar dtypes (e.g., np.uint64, np.int32) to maintain
    ROS type precision in the output data.

    Note: Do NOT call .item() on numpy scalars as this converts them back to
    Python types and loses dtype information.
    """
    # Handle None/NaN
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    # Handle numpy scalars (from type conversion) - preserve dtype
    if isinstance(value, np.generic):
        # Keep numpy scalar wrapped in list to preserve dtype
        # e.g., [np.float32(3.14)] not [3.14] (Python float)
        return [value]

    # Handle 0-dimensional numpy arrays
    if isinstance(value, np.ndarray) and value.ndim == 0:
        # Extract scalar value from 0-d array
        # Note: 0-d arrays have already lost their context, so .item() is acceptable here
        return [value.item()]

    # Already an array or list
    if isinstance(value, (np.ndarray, list)):
        # Skip empty arrays
        if len(value) == 0:
            return None
        return value

    # Scalar value - wrap in list
    return [value]


def expand_array_columns_vertically(df):
    """Expands all columns containing arrays vertically, creating new rows for each array element.

    Preservation strategy:
    - Converts scalars to [scalar]
    - Keeps non-empty arrays as-is
    - Skips empty arrays []
    - Concatenates all values vertically across all rows
    - One observation per cell (no nested arrays)
    """
    # Early exit for empty DataFrame
    if df.empty:
        return df

    array_columns = []
    non_array_columns = []
    for col in df.columns:
        if is_array_column(df[col]):
            array_columns.append(col)
        else:
            non_array_columns.append(col)

    logger.debug(f"Found array columns: {array_columns}")
    if not array_columns:
        return df

    # Collect all array values across all rows for each array column
    # This implements the concatenation strategy: [a, b], [], [c] -> [a, b, c]
    all_arrays = {col: [] for col in array_columns}
    non_array_values = {col: [] for col in non_array_columns}

    for idx, row in df.iterrows():
        # Normalize each array column value
        normalized_arrays = {}
        for col in array_columns:
            normalized = _normalize_to_array(row[col])
            normalized_arrays[col] = normalized

        # Check if all array columns are None (should skip this row entirely)
        if all(arr is None for arr in normalized_arrays.values()):
            logger.debug(f"Skipping row {idx}: all array columns are empty")
            continue

        # Get the maximum length among non-None arrays in this row
        lengths = [len(arr) for arr in normalized_arrays.values() if arr is not None]
        if not lengths:
            continue

        max_length = max(lengths)

        # Validate: all non-None arrays should have the same length
        # This is a data integrity check - inconsistent array lengths indicate
        # corrupted data or transmission errors that must be investigated
        if not all(length == max_length for length in lengths):
            raise ValueError(
                f"Data integrity error at row {idx}: Array columns have inconsistent lengths {lengths}. "
                f"All nested arrays in a ROS message must have the same length. "
            )

        # Concatenate this row's arrays to the master list
        for col in array_columns:
            if normalized_arrays[col] is not None:
                all_arrays[col].extend(normalized_arrays[col])
            else:
                # If one column is None but others aren't, pad with NaN
                all_arrays[col].extend([np.nan] * max_length)

        # Repeat non-array values to match array length
        for col in non_array_columns:
            non_array_values[col].extend([row[col]] * max_length)

    # Build the result DataFrame
    result_data = {}
    for col in non_array_columns:
        result_data[col] = non_array_values[col]
    for col in array_columns:
        result_data[col] = all_arrays[col]

    result_df = pd.DataFrame(result_data)

    # Preserve dtypes from original DataFrame
    # When repeating values, pandas may upcast types (e.g., int32 -> int64)
    # We need to explicitly restore the original dtypes
    for col in df.columns:
        if col in result_df.columns:
            original_dtype = df[col].dtype
            # Only apply dtype if it's not object (which might contain arrays/complex types)
            if original_dtype != "object" and result_df[col].dtype != original_dtype:
                try:
                    result_df[col] = result_df[col].astype(original_dtype)
                except (ValueError, TypeError):
                    # If conversion fails, keep the inferred dtype
                    logger.debug(
                        f"Could not preserve dtype {original_dtype} for column {col}"
                    )

    logger.info(f"Expanded shape from {df.shape} to {result_df.shape}")
    return result_df


# Original custom schema parsing implementation
# Replaced with library-based version below for better reliability and maintainability
# def parse_ros_message_definition(
#     definition: str | bytes, _all_sections: list[str] | None = None
# ) -> dict[str, Any]:
#     """Parse a ROS message definition into a dictionary describing the message structure.
#
#     Args:
#         definition: The ROS message definition string or bytes
#         _all_sections: Internal parameter - full list of all schema sections for recursive lookups
#
#     Returns:
#         Dictionary mapping field names to their specifications
#     """
#     if isinstance(definition, bytes):
#         try:
#             definition_str = definition.decode("utf-8")
#         except UnicodeDecodeError:
#             try:
#                 definition_str = definition.decode("ascii")
#             except UnicodeDecodeError:
#                 definition_str = definition.decode("latin-1")
#     else:
#         definition_str = definition
#
#     # ROS message constant definitions to skip (not actual message fields)
#     ROS_CONSTANTS = {
#         "DEBUG=10",
#         "INFO=20",
#         "WARN=30",
#         "ERROR=40",
#         "FATAL=50",
#     }
#
#     message_spec: dict[str, dict] = {}
#
#     # Split into sections on first call, or use existing sections
#     if _all_sections is None:
#         sections = definition_str.split(
#             "================================================================================"
#         )
#         _all_sections = sections  # Preserve all sections for recursive calls
#     else:
#         sections = [definition_str]  # For recursive calls, only parse main section
#
#     main_section = sections[0].strip()
#
#     for line in main_section.split("\n"):
#         line = line.strip()
#         if not line or line.startswith("#"):
#             continue
#
#         parts = line.split()
#         if len(parts) >= 2:
#             field_type, field_name = parts[0], parts[1]
#
#             # Skip known ROS logging level constants
#             if field_name in ROS_CONSTANTS:
#                 logger.debug(f"Skipping ROS constant: {field_name}")
#                 continue
#
#             is_array = field_type.endswith("[]")
#             if is_array:
#                 field_type = field_type[:-2]
#
#             default_value = None
#             if len(parts) >= 3:
#                 try:
#                     raw_default = parts[2].strip('"')
#                     if field_type == "bool":
#                         default_value = raw_default.lower() == "true"
#                     elif field_type == "string":
#                         default_value = raw_default
#                     elif field_type.startswith("float"):
#                         default_value = float(raw_default)
#                     elif field_type.startswith("int"):
#                         default_value = int(raw_default)
#                 except:
#                     pass
#
#             message_spec[field_name] = {
#                 "type": field_type,
#                 "is_array": is_array,
#                 "default": default_value,
#             }
#
#             # For nested types (e.g., std_msgs/Header, builtin_interfaces/Time),
#             # search ALL available sections to find the type definition
#             if "/" in field_type:
#                 type_name = field_type.split("/")[-1]
#                 for section in _all_sections[
#                     1:
#                 ]:  # Use all sections, not just local ones
#                     if f"MSG: {field_type}" in section:
#                         # Pass all sections to recursive call so deeply nested types can be found
#                         nested_fields = parse_ros_message_definition(
#                             section, _all_sections
#                         )
#                         message_spec[field_name]["fields"] = nested_fields
#                         break
#
#     return message_spec


def parse_ros_message_definition(
    definition: str | bytes, _all_sections: list[str] | None = None
) -> dict[str, Any]:
    """Parse a ROS message definition into a dictionary describing the message structure.

    This implementation uses the MCAP library's built-in rosidl_adapter parser
    instead of custom regex-based parsing for better reliability and maintainability.

    Args:
        definition: The ROS message definition string or bytes
        _all_sections: Internal parameter - kept for API compatibility but not used
                      (the library parser handles multi-section schemas internally)

    Returns:
        Dictionary mapping field names to their specifications, with structure:
        {
            "field_name": {
                "type": "type_name",
                "is_array": bool,
                "default": value or None,
                "fields": {...}  # For nested types only
            }
        }
    """
    from mcap_ros2._dynamic import _for_each_msgdef
    from mcap_ros2._vendor.rosidl_adapter.parser import MessageSpecification

    # Convert bytes to string if needed
    if isinstance(definition, bytes):
        try:
            definition_str = definition.decode("utf-8")
        except UnicodeDecodeError:
            try:
                definition_str = definition.decode("ascii")
            except UnicodeDecodeError:
                definition_str = definition.decode("latin-1")
    else:
        definition_str = definition

    # Extract schema name from the first line or use a default
    # The schema name is needed by _for_each_msgdef but in our use case
    # we only care about the main (first) message definition

    # Use a placeholder schema name - the library will parse all sections
    # and we'll just take the first one (which is the main message)
    schema_name = "parsed/Message"

    # Use the MCAP library's parser to handle all sections
    all_msgdefs: dict[str, MessageSpecification] = {}
    first_schema_name = None

    def collect_msgdef(
        cur_schema_name: str, short_name: str, msgdef: MessageSpecification
    ):
        """Collect all message definitions found in the schema."""
        nonlocal first_schema_name
        if first_schema_name is None:
            first_schema_name = cur_schema_name
        all_msgdefs[cur_schema_name] = msgdef
        all_msgdefs[short_name] = msgdef

    # Parse the schema using the library's built-in parser
    _for_each_msgdef(schema_name, definition_str, collect_msgdef)

    # Get the main message definition (the first one parsed)
    # The first message parsed is always the main/top-level message
    if not all_msgdefs:
        # Empty schema
        return {}

    if first_schema_name and first_schema_name in all_msgdefs:
        main_msgdef = all_msgdefs[first_schema_name]
    else:
        # Fallback to first in dict
        main_msgdef = next(iter(all_msgdefs.values()))

    def field_to_dict(field) -> dict[str, Any]:
        """Convert a Field object to our dictionary format."""
        # Build the type string (with package name if present)
        if field.type.pkg_name:
            type_str = f"{field.type.pkg_name}/{field.type.type}"
        else:
            type_str = field.type.type

        spec = {
            "type": type_str,
            "is_array": field.type.is_array,
            "default": field.default_value,
        }

        # Recursively handle nested types
        if not field.type.is_primitive_type():
            nested_key = f"{field.type.pkg_name}/{field.type.type}"
            if nested_key in all_msgdefs:
                nested_msgdef = all_msgdefs[nested_key]
                spec["fields"] = {
                    nested_field.name: field_to_dict(nested_field)
                    for nested_field in nested_msgdef.fields
                }

        return spec

    # Convert all fields to our dictionary format
    return {field.name: field_to_dict(field) for field in main_msgdef.fields}


class MessageProcessor:
    def __init__(self, schema: dict[str, Any]):
        self.schema = schema
        self.messages: list[dict[str, Any]] = []

        self.ros_type_to_numpy_type_map = {
            "float32": np.float32,
            "float64": np.float64,
            "int8": np.int8,
            "int16": np.int16,
            "int32": np.int32,
            "int64": np.int64,
            "uint8": np.uint8,
            "uint16": np.uint16,
            "uint32": np.uint32,
            "uint64": np.uint64,
        }

    def _apply_type_conversion(self, value: Any, ros_type: str) -> Any:
        """Apply numpy type conversion if the ROS type is in the conversion map.

        Args:
            value: The value to convert (can be scalar or array)
            ros_type: The ROS type string (e.g., "float64", "uint32")

        Returns:
            For scalars: numpy scalar with correct dtype (e.g., np.int32(5))
            For arrays: numpy array with correct dtype
            Otherwise: original value unchanged
        """
        if ros_type in self.ros_type_to_numpy_type_map:
            numpy_dtype = self.ros_type_to_numpy_type_map[ros_type]

            # Check if value is already an array/list
            if isinstance(value, (list, np.ndarray)):
                return np.array(value, dtype=numpy_dtype)
            else:
                # For scalars, return numpy scalar type (not wrapped in array)
                # This ensures pandas can properly infer column dtype
                return numpy_dtype(value)
        return value

    def _extract_nested_fields(
        self, msg_obj: Any, field_spec: dict[str, Any]
    ) -> dict[str, Any]:
        """Recursively extract fields from nested ROS message types.

        This method handles complex message types like std_msgs/Header or
        builtin_interfaces/Time by walking through their nested field structure
        defined in the schema and extracting values from the message object.

        Args:
            msg_obj: The ROS message object to extract fields from
            field_spec: The schema specification for this field (must have "fields" key)

        Returns:
            Dictionary mapping field names to their values (with type conversion applied)

        Example:
            For a header field with stamp (builtin_interfaces/Time), this extracts:
            {
                "sec": 1234567890,
                "nanosec": 123456789,
                "frame_id": "base_link"
            }
        """
        result = {}

        if "fields" not in field_spec:
            return result

        nested_fields = field_spec["fields"]

        for nested_name, nested_spec in nested_fields.items():
            # Skip MSG: marker entries (these are just schema metadata)
            if nested_spec.get("type", "").startswith("MSG:"):
                continue

            try:
                nested_value = getattr(msg_obj, nested_name)
                nested_type = nested_spec.get("type")

                # Recursively handle nested types (e.g., stamp is builtin_interfaces/Time)
                if "fields" in nested_spec:
                    sub_fields = self._extract_nested_fields(nested_value, nested_spec)
                    result.update(sub_fields)
                else:
                    result[nested_name] = self._apply_type_conversion(
                        nested_value, nested_type
                    )

            except (AttributeError, RuntimeError) as e:
                logger.warning(f"Failed to extract nested field '{nested_name}': {e}")

        return result

    def process_message(self, msg: Any) -> None:
        """Process a ROS message and extract all fields according to the schema.

        Handles both simple fields and nested message types (like std_msgs/Header).
        Applies numpy type conversion based on the schema's ROS type definitions.
        """
        message_dict = {}

        for field_name, field_spec in self.schema.items():
            try:
                value = getattr(msg, field_name)
                field_type = field_spec.get("type")

                # Check if this field has nested structure (e.g., header, pose, etc.)
                if "fields" in field_spec:
                    # Extract nested fields and flatten into message_dict
                    nested_data = self._extract_nested_fields(value, field_spec)
                    message_dict.update(nested_data)
                else:
                    # Simple field - apply type conversion
                    message_dict[field_name] = self._apply_type_conversion(
                        value, field_type
                    )

            except (AttributeError, RuntimeError) as e:
                logger.warning(f"Failed to get field '{field_name}': {e}")

        self.messages.append(message_dict)

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.messages)
        if "sec" in df.columns and "nanosec" in df.columns:
            df["timestamp"] = df["sec"] + df["nanosec"] * 1e-9

        if "stamp" in df.columns:
            df["sec"] = df["stamp"].apply(lambda t: t.sec)
            df["nanosec"] = df["stamp"].apply(lambda t: t.nanosec)

            df = df.drop(columns=["stamp"])

            df["timestamp"] = df["sec"] + df["nanosec"] * 1e-9

        return df
