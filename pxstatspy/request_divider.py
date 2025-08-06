import math
from itertools import product as iter_product
from typing import Dict, List, Optional, Tuple

RequestConfig = Dict[str, List[str]]
Partition = Tuple[int, int]


# -----------------------------------------------------------------------------
#  Motivation and context
#  ----------------------
#  Many statistical APIs impose a maximum number of rows ("cells") that can be
#  returned in a single request.  A selection that crosses several dimensions
#  (e.g. sex × year × country) may easily exceed this limit, requiring the
#  caller to break the original query into several smaller requests.
#
#  The functions below do exactly that:
#      * They find batch sizes for every dimension such that the product of the
#        batch sizes (the rows requested per call) never exceeds *cell_limit*.
#      * They choose those batch sizes so that the **total number of API calls
#        is minimized**.
#      * They finally return concrete request configurations (Cartesian
#        products of the batched values) that cover the entire original
#        selection.
#
#  The algorithm is fully deterministic and does not rely on external state.
# -----------------------------------------------------------------------------


def _get_partitions(
    counts: List[int],
    value_limit: int,
) -> List[List[Partition]]:
    """Enumerate feasible ``(num_batches, batch_size)`` pairs for each variable.

    For a variable with *n* distinct values we consider cutting the list into
    equally‑sized *batches* (the last batch may be smaller).  For every
    feasible number of batches we keep **only the smallest** corresponding
    batch size, because using larger batch sizes would always increase the
    product and never decrease the request count.

    Args:
        counts:         Number of distinct values per variable.
        value_limit:    The maximum number of values allowed for each variable.

    Returns:
        A list whose *i*‑th element contains the feasible ``(num_batches,
        batch_size)`` pairs for the *i*‑th variable.
    """
    options: List[List[Partition]] = []

    for n_values in counts:
        # Mapping: num_batches -> smallest feasible batch_size
        best_for_batches: dict[int, int] = {}

        # Iterate *descending* so that the first time we hit a particular
        # num_batches it is guaranteed to be the **smallest** batch size that
        # yields that many batches.
        for batch_size in range(min(n_values, value_limit), 0, -1):
            num_batches = math.ceil(n_values / batch_size)
            best_for_batches[num_batches] = batch_size

        options.append(list(best_for_batches.items()))

    return options


# -----------------------------------------------------------------------------
#  _find_optimal_combination
#  -------------------------
#  Core search routine.  It looks at *all* variables simultaneously and finds
#  exactly one batch size for every variable such that the product of the chosen
#  sizes never exceeds *cell_limit* while the resulting total number of API
#  requests (= product of the "num_batches" selections) is minimal.
# -----------------------------------------------------------------------------


def _find_optimal_combination(
    value_counts: List[int], cell_limit: int, value_limit: int
) -> List[int]:
    """
    Finds the combination of batch sizes that minimizes API requests.

    This function solves the core optimization problem. It explores combinations
    of batching strategies for each variable to find the one that results in
    the fewest total API calls while respecting the cell `limit`.

    Args:
        value_counts: A list of value counts for each variable.
        limit: The maximum number of cells to request.
        value_limit: The maximum number of values for each variable.

    Returns:
        A list of optimal batch sizes, in the same order as the input `value_counts`.
    """
    total_cells = math.prod(value_counts)
    if total_cells <= cell_limit:
        return value_counts

    # Track original indices to correctly reorder the final result.
    # Sorting by size descending is a heuristic that makes pruning more effective,
    # as it deals with the most constrained variables first.
    indexed_counts = sorted(enumerate(value_counts), key=lambda x: x[1], reverse=True)
    original_indices = [i for i, _ in indexed_counts]
    sorted_counts = [n for _, n in indexed_counts]

    all_partitions = _get_partitions(sorted_counts, value_limit)

    best_combo: List[Partition] = []
    min_requests = float("inf")
    # The theoretical minimum number of requests provides a powerful early-exit condition.
    lower_request_bound = math.ceil(total_cells / cell_limit)

    def search(
        combo_so_far: List[Partition],
        index: int,
        current_requests: int,
        current_cells: int,
    ):
        """Recursively search for the best combination with pruning."""
        nonlocal best_combo, min_requests

        if current_requests >= min_requests or current_cells > cell_limit:
            return

        if index == len(sorted_counts):
            if current_requests < min_requests:
                min_requests = current_requests
                best_combo = combo_so_far
            return

        for num_batches, batch_size in all_partitions[index]:
            search(
                combo_so_far + [(num_batches, batch_size)],
                index + 1,
                current_requests * num_batches,
                current_cells * batch_size,
            )
            # If we've already found a solution that hits the theoretical minimum,
            # we can stop searching immediately.
            if min_requests == lower_request_bound:
                return

    search([], 0, 1, 1)

    # Reconstruct the result in the original variable order.
    # This is more robust and clearer than the previous implementation.
    final_batch_sizes = [0] * len(value_counts)
    # The batch size is the second element of the partition tuple.
    sorted_batch_sizes = [p[1] for p in best_combo]
    for original_idx, batch_size in zip(original_indices, sorted_batch_sizes):
        final_batch_sizes[original_idx] = batch_size

    return final_batch_sizes


def _batch_variable_values(
    variables: Dict[str, List[str]], batch_sizes: Dict[str, int]
) -> Dict[str, List[List[str]]]:
    """
    Splits variable values into batches based on the calculated sizes.

    Args:
        variables: The original dictionary of variables and their values.
        batch_sizes: A dictionary mapping variable names to their optimal batch size.

    Returns:
        A dictionary mapping variable names to their batched values.
    """
    batched_dict = {}
    for name, values in variables.items():
        batch_size = batch_sizes.get(name)
        # A batch size should always be a positive integer.
        if not batch_size:
            continue

        batched_dict[name] = [
            values[i : i + batch_size] for i in range(0, len(values), batch_size)
        ]

    return batched_dict


def _check_input(
    selection: Dict[str, List[str]],
    cell_limit: int,
    value_limit: int | None = None,
) -> None:
    """Validate input parameters for the request divider."""
    if not isinstance(selection, dict):
        raise TypeError(f"Selection must be a dictionary.")
    if not all(
        isinstance(k, str) and isinstance(v, list) for k, v in selection.items()
    ):
        raise TypeError("Selection keys must be strings and values must be lists.")
    if not isinstance(cell_limit, int) or cell_limit <= 0:
        raise ValueError("Cell limit must be a positive integer.")
    if value_limit is not None and (
        not isinstance(value_limit, int) or value_limit <= 0
    ):
        raise ValueError("Value limit must be a positive integer or None.")

    # check that cell_limit is not smaller than value_limit
    if value_limit is not None and cell_limit < value_limit:
        raise ValueError("Cell limit must be greater than or equal to the value limit.")


def divide_requests(
    selection: Dict[str, List[str]],
    cell_limit: int,
    value_limit: Optional[int] = None,
) -> List[RequestConfig]:
    """
    Divides a user selection into multiple smaller requests based on API limits.

    This function calculates the most efficient way to query an API that has
    a limit on the total number of data cells (e.g., countries * years)
    that can be requested at once.

    Args:
        selection: A dictionary with dimension names as keys and lists of
                   selected values as values.
        cell_limit: The maximum number of cells to request in a single API call.
        value_limit: An optional limit on the number of values per dimension.
                     If None, it defaults to the `cell_limit`.

    Returns:
        A list of request configurations, where each configuration is a
        dictionary ready to be used in an API call.
    """
    if not selection:
        return []

    # Validate input parameters.
    _check_input(selection, cell_limit, value_limit)

    # REFACTOR: Consolidated the early exit check here.
    total_cells = math.prod(len(v) for v in selection.values())
    if total_cells <= cell_limit:
        return [selection]

    # If `value_limit` is not provided, it can't be larger than the cell limit.
    effective_value_limit = value_limit or cell_limit

    # We preserve the original order and let `_find_optimal_combination` handle its own sorting internally.
    variable_names = list(selection.keys())
    value_counts = [len(selection[name]) for name in variable_names]

    optimal_batch_sizes = _find_optimal_combination(
        value_counts, cell_limit, effective_value_limit
    )

    # Create a mapping from name to size for easier lookup.
    batch_sizes_by_name = dict(zip(variable_names, optimal_batch_sizes))

    batched_values = _batch_variable_values(selection, batch_sizes_by_name)

    # Use itertools.product to create all combinations of the value batches.
    # The keys in `batched_values` are already in the desired order.
    request_configs = []
    batch_lists = batched_values.values()

    for combination in iter_product(*batch_lists):
        request_configs.append(dict(zip(batched_values.keys(), combination)))

    return request_configs


if __name__ == "__main__":
    import json

    user_selection = {
        "year": ["2019", "2020", "2021", "2022", "2023"],  # 5
        "sex": ["total", "women", "men"],  # 3
        "country of birth": ["Norway", "Finland", "Sweden", "Denmark"],  # 4
    }
    # Total cells = 3 * 4 * 5 = 60
    row_limit = 13
    # Theoretical min requests = ceil(60 / 13) = 5

    requests = divide_requests(user_selection, row_limit)
    print(
        f"Original selection would require {math.prod(len(v) for v in user_selection.values())} cells."
    )
    print(
        f"Divided into {len(requests)} requests to stay under the {row_limit} cell limit.\n"
    )
    print(json.dumps(requests, indent=2, ensure_ascii=False))
