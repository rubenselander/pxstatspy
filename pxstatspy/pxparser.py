from typing import Any, Dict, List
from .request_divider import divide_requests

ValueCodes = Dict[str, List[str]]
ChunkedValueCodes = List[ValueCodes]


class SelectionMatcher:
    """Handles pattern matching and expansion for PxAPI selection expressions"""

    @staticmethod
    def expand_expression(expr: str, all_codes: List[str]) -> List[str]:
        """
        Expand a selection expression into actual value codes

        Args:
            expr: Selection expression (e.g., 'TOP(5)', 'RANGE(2020,2023)')
            all_codes: List of all possible value codes

        Returns:
            List of value codes matching the expression
        """
        # Extract function name and arguments
        func = expr[: expr.find("(")].upper()
        args = expr[expr.find("(") + 1 : expr.find(")")].split(",")
        args = [arg.strip() for arg in args]

        if func == "TOP":
            n = int(args[0])
            offset = int(args[1]) if len(args) > 1 else 0
            return all_codes[offset : offset + n]

        elif func == "BOTTOM":
            n = int(args[0])
            offset = int(args[1]) if len(args) > 1 else 0
            return all_codes[-(n + offset) : -offset if offset else None]

        elif func == "RANGE":
            if len(args) != 2:
                raise ValueError(
                    f"RANGE expression requires 2 arguments, got {len(args)}"
                )
            start, end = args
            start_idx = all_codes.index(start)
            end_idx = all_codes.index(end)
            return all_codes[start_idx : end_idx + 1]

        elif func == "FROM":
            if len(args) != 1:
                raise ValueError(
                    f"FROM expression requires 1 argument, got {len(args)}"
                )
            start_idx = all_codes.index(args[0])
            return all_codes[start_idx:]

        elif func == "TO":
            if len(args) != 1:
                raise ValueError(f"TO expression requires 1 argument, got {len(args)}")
            end_idx = all_codes.index(args[0])
            return all_codes[: end_idx + 1]

        else:
            raise ValueError(f"Unknown selection expression: {func}")

    @staticmethod
    def get_matching_codes(patterns: List[str], all_codes: List[str]) -> List[str]:
        """
        Get all codes that match any of the given patterns

        Args:
            patterns: List of selection patterns
            all_codes: List of all possible value codes

        Returns:
            List of matching value codes (deduplicated)
        """
        matched_codes = []

        for pattern in patterns:
            # Handle wildcard "*" directly
            if pattern == "*":
                return all_codes

            # Handle other wildcards and question marks
            if "*" in pattern or "?" in pattern:
                import fnmatch

                matches = [code for code in all_codes if fnmatch.fnmatch(code, pattern)]
                matched_codes.extend(matches)
                continue

            # Handle selection expressions
            if "(" in pattern and ")" in pattern:
                try:
                    matches = SelectionMatcher.expand_expression(pattern, all_codes)
                    matched_codes.extend(matches)
                    continue
                except ValueError:
                    pass

            # Direct match
            if pattern in all_codes:
                matched_codes.append(pattern)

        # Remove duplicates while preserving order
        return list(dict.fromkeys(matched_codes))


# def extract_variable_types(metadata_stat: Dict[str, Any]) -> Dict[str, str]:
#     """
#     Extract variable types (roles) from JSON-stat2 metadata

#     Args:
#         metadata_stat: JSON-stat2 metadata dictionary

#     Returns:
#         Dictionary mapping variable IDs to their detected types/roles
#         (one of: TimeVariable, GeographicalVariable, ContentsVariable, RegularVariable)
#     """
#     variable_types = {}
#     for var_id in metadata_stat["id"]:
#         # Try to determine variable type from role or naming conventions
#         var_type = PxVariables.TYPE_REGULAR  # Default type

#         if "role" in metadata_stat:
#             # Use the role attribute if available (standard JSON-stat2 approach)
#             if var_id in metadata_stat["role"].get("time", []):
#                 var_type = PxVariables.TYPE_TIME
#             elif var_id in metadata_stat["role"].get("metric", []):
#                 var_type = PxVariables.TYPE_CONTENTS
#             elif var_id in metadata_stat["role"].get("geo", []):
#                 var_type = PxVariables.TYPE_GEOGRAPHICAL
#         else:
#             # Fall back to variable name patterns if role isn't defined
#             var_id_lower = var_id.lower()

#             # Check if matches exact variable names
#             if var_id == PxVariables.TIME:
#                 var_type = PxVariables.TYPE_TIME
#             elif var_id == PxVariables.CONTENTS:
#                 var_type = PxVariables.TYPE_CONTENTS
#             elif var_id == PxVariables.REGION:
#                 var_type = PxVariables.TYPE_GEOGRAPHICAL
#             # Check if contains any alternative names
#             elif any(alt in var_id_lower for alt in PxVariables.TIME_ALTERNATIVES):
#                 var_type = PxVariables.TYPE_TIME
#             elif any(alt in var_id_lower for alt in PxVariables.CONTENTS_ALTERNATIVES):
#                 var_type = PxVariables.TYPE_CONTENTS
#             elif any(alt in var_id_lower for alt in PxVariables.REGION_ALTERNATIVES):
#                 var_type = PxVariables.TYPE_GEOGRAPHICAL

#         # Store the detected type
#         variable_types[var_id] = var_type

#     return variable_types


def chunk_value_codes(
    value_codes: ValueCodes,
    metadata_stat: Dict[str, Any],
    max_cells: int = 150000,
) -> ChunkedValueCodes:
    """
    Parse, chunk and validate value codes against metadata.

    Args:
        value_codes: Dictionary mapping variable IDs to lists of value codes or selection expressions
        metadata_stat: JSON-stat2 metadata dictionary
        max_cells: Maximum number of data cells per request (default: 150000)

    Returns:
        ParsedSelection instance with validated and processed value codes
    """
    if not isinstance(value_codes, dict):
        raise ValueError(f"Value codes must be a dictionary, got {type(value_codes)}")

    dimension = metadata_stat.get("dimension", {})
    all_value_codes = {
        dim_id: list(dim["category"]["index"].keys())
        for dim_id, dim in dimension.items()
    }
    dim_elimination = {
        dim_id: dim.get("extension", {}).get("elimination", False)
        for dim_id, dim in dimension.items()
    }

    # # Placeholder for potential variable type handling
    # # Could be controlled through optional input parameter

    # variable_types = extract_variable_types(metadata_stat)
    # non_chunkable_vars = {
    # var_id for var_id, var_type in variable_types.items()
    # if var_type in {PxVariables.TYPE_TIME, PxVariables.TYPE_CONTENTS}
    # }

    parsed_value_codes = {}

    for var_id, all_codes in all_value_codes.items():
        input_codes = value_codes.get(var_id, [])
        if input_codes and len(input_codes) > 0:
            matched_codes = SelectionMatcher.get_matching_codes(input_codes, all_codes)
        else:
            matched_codes = []

        elimination = dim_elimination.get(var_id, False)

        if not matched_codes and elimination:
            # If no codes matched and variable can be eliminated, skip it
            continue
        elif not matched_codes:
            # If no codes matched and variable cannot be eliminated, use all codes
            matched_codes = all_codes

        parsed_value_codes[var_id] = matched_codes

    # Remove non-chunkable variables and
    # divide max_cells by the corresponding number of matched codes
    # (These will not be chunked)
    # constant_value_codes = {}
    # for var_id in non_chunkable_vars:
    #    if var_id in parsed_value_codes:
    #        constant_value_codes[var_id] = parsed_value_codes.pop(var_id)
    # if constant_value_codes:
    #   total_constant_cells = math.prod(
    #       len(codes) for codes in constant_value_codes.values()
    #   )
    #   if total_constant_cells > max_cells:
    #       raise PxAPIError(
    #           f"Non-chunkable variables would return {total_constant_cells} cells, "
    #           f"which exceeds the maximum of {max_cells}. "
    #           "Consider chunking all variables or reducing selections."
    #       )
    #  # Divide max_cells by the number of constant codes to accommodate them
    #   max_cells //= total_constant_cells

    # Divide into chunks
    chunked_value_codes = divide_requests(parsed_value_codes, cell_limit=max_cells)

    # # add back constant value codes to each chunk
    # for chunk in chunked_value_codes:
    #     chunk.update(constant_value_codes)

    return chunked_value_codes
