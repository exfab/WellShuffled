"""Utility scripts for `wellshuffled`."""

import csv
import re

import click
import numpy as np


def normalize_well_key(well: str) -> str:
    """Normalize the case of an ASCII well position without Unicode case-mapping.

    ``str.upper()`` folds some non-ASCII letters onto ASCII ones (e.g. 'ı' becomes
    'I'), which would let a well outside A-Z pass validation as a real one. Non-ASCII
    input is returned unchanged so that well_to_index rejects it.

    Parameters
    ----------
    well : str
        The well position to normalize.

    Returns
    -------
    str
        The upper-cased well if it is ASCII, otherwise the original unchanged.
    """
    return well.upper() if well.isascii() else well


def letters_to_row_index(letters: str) -> int:
    """Convert a row label into its 0-based row index.

    Rows are labelled alphabetically and continue into multiple letters beyond 'Z',
    matching 1536-well plates where row 27 is 'AA' (e.g. 'A'..'Z', 'AA'..'AF').

    Parameters
    ----------
    letters : str
        The row label (e.g., 'A', 'H', 'AA', 'AF'). Case-insensitive.

    Returns
    -------
    int
        The 0-based row index (e.g., 0, 7, 26, 31).

    Raises
    ------
    ValueError
        If the label is empty or contains anything other than ASCII letters A-Z.
    """
    # isalpha() alone accepts any Unicode letter, so restrict to ASCII A-Z.
    if not letters or not (letters.isascii() and letters.isalpha()):
        raise ValueError(f"Invalid row label: '{letters}'. Expected letters A-Z only.")

    index = 0
    for letter in letters.upper():
        index = index * 26 + (ord(letter) - ord("A") + 1)
    return index - 1


def row_index_to_letters(row_index: int) -> str:
    """Convert a 0-based row index into its row label.

    Rows are labelled alphabetically and continue into multiple letters beyond 'Z',
    matching 1536-well plates where row 27 is 'AA' (e.g. 'A'..'Z', 'AA'..'AF').

    Parameters
    ----------
    row_index : int
        The 0-based row index (e.g., 0, 7, 26, 31).

    Returns
    -------
    str
        The row label (e.g., 'A', 'H', 'AA', 'AF').

    Raises
    ------
    ValueError
        If the row index is negative.
    """
    if row_index < 0:
        raise ValueError(f"Invalid row index: {row_index}. Must be non-negative.")

    letters = ""
    index = row_index + 1
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def convert_well_number_to_position(well_number: int, plate_dims: tuple[int, int]) -> str:
    """Convert a 1-based column-major well number to an alphanumeric well position.

    Parameters
    ----------
    well_number : int
        The 1-based column-major well number.
    plate_dims : tuple[int, int]
        The dimensions of the plate (rows, cols).

    Returns
    -------
    str
        The alphanumeric well position (e.g., 'A1').

    Raises
    ------
    ValueError
        If the well number is out of bounds for the given plate dimensions.
    """
    rows, cols = plate_dims
    if not (1 <= well_number <= rows * cols):
        raise ValueError(f"Well number {well_number} is out of bounds for a {rows}x{cols} plate.")

    col_index = (well_number - 1) // rows
    row_index = (well_number - 1) % rows

    row_letter = row_index_to_letters(row_index)
    col_number = col_index + 1

    return f"{row_letter}{col_number}"


def well_to_index(well: str, plate_dims: tuple[int, int]) -> tuple[int, int]:
    """Convert a standard well designation to a 0-based (row, col) index.

    Parameters
    ----------
    well : str
        The standard well designation (e.g., 'A1', 'H12').
    plate_dims : tuple[int, int]
        The dimensions of the plate (rows, cols).

    Returns
    -------
    tuple[int, int]
        The 0-based (row, col) index.

    Raises
    ------
    ValueError
        If the well designation is invalid or out of bounds.
    """
    rows, cols = plate_dims

    if well.isdigit():
        well = convert_well_number_to_position(int(well), plate_dims)

    # Split the leading row label from the trailing column number. Rows may span
    # multiple letters (e.g. 'AA1'), so match a run of letters rather than one char.
    # Character classes are explicit ASCII so non-ASCII digits are rejected too (\d
    # would otherwise accept e.g. Arabic-Indic digits, which int() also accepts).
    match = re.match(r"^([A-Za-z]+)([0-9]+)$", well)
    if not match:
        raise ValueError(f"Invalid well designation: {well}")

    # Determine row index
    row_index = letters_to_row_index(match.group(1))

    # Determine column index
    col_index = int(match.group(2)) - 1

    # Validation
    if not (0 <= row_index < rows and 0 <= col_index < cols):
        max_row_letter = row_index_to_letters(rows - 1)
        raise ValueError(
            f"Well {well} is outside plate dimensions ({rows}x{cols}). "
            f"Max well is {max_row_letter}{cols}."
        )

    return row_index, col_index


def convert_position_to_well_number(well_position: str, plate_dims: tuple[int, int]) -> str:
    """Convert an alphanumeric well position to its 1-based column-major number.

    Parameters
    ----------
    well_position : str
        The alphanumeric well position (e.g., 'A1').
    plate_dims : tuple[int, int]
        The dimensions of the plate (rows, cols).

    Returns
    -------
    str
        The 1-based column-major well number.

    Raises
    ------
    ValueError
        If the well position is invalid or out of bounds.
    """
    rows, cols = plate_dims
    raw_position = str(well_position)

    # Match before upper-casing: Unicode case-mapping can turn non-ASCII letters into
    # ASCII ones (e.g. 'ß' -> 'SS', 'ı' -> 'I'), which would otherwise be silently
    # accepted as a multi-letter row. Character classes are explicit ASCII so that
    # non-ASCII digits are rejected as well.
    match = re.match(r"^([A-Za-z]+)([0-9]+)$", raw_position)
    well_position = raw_position.upper()

    if not match:
        # If input is already numeric, return it (useful if fixed map used numbers).
        # isdigit() alone also accepts non-ASCII digits, so require ASCII here too.
        if raw_position.isascii() and raw_position.isdigit():
            return well_position
        raise ValueError(f"Invalid well position format: '{well_position}'")

    row_letter = match.group(1)
    col_str = match.group(2)

    num_rows, num_cols = rows, cols

    row_index = letters_to_row_index(row_letter)
    col_index = int(col_str) - 1

    if not (0 <= row_index < num_rows and 0 <= col_index < num_cols):
        max_row_letter = row_index_to_letters(num_rows - 1)
        raise ValueError(
            f"Well {well_position} is outside plate dimensions ({num_rows}x{num_cols}). Max well is {max_row_letter}{num_cols}."
        )

    well_number = (col_index * num_rows) + (row_index + 1)

    return str(well_number)


def load_control_map_from_csv(filename: str) -> dict[str, str]:
    """Load a control map from a two-column CSV file.

    Parameters
    ----------
    filename : str
        Path to the CSV file. The file should have two columns: Well and Sample ID.

    Returns
    -------
    dict[str, str]
        A dictionary mapping well position string (e.g., 'A1') to sample ID.

    Raises
    ------
    ValueError
        If the CSV file is malformed.
    """
    # The map is temporarily stored as {Well: Sample ID} string-to-string
    control_map = {}

    with open(filename, "r", newline="") as csv_file:
        # Use csv.reader to handle different delimiters/quoting if needed
        reader = csv.reader(csv_file)

        # Skip header row (assuming the first row is a header)
        try:
            next(reader)
        except StopIteration:
            # Handle empty file case, though the CLI should catch this
            return {}

        for i, row in enumerate(reader, start=2):
            if not row:
                continue

            # Expecting exactly two columns: Well and Sample ID
            if len(row) < 2:
                raise ValueError(
                    f"Fixed map file '{filename}' row {i} is missing data. Expected 'Well,Sample ID'."
                )

            well_pos = normalize_well_key(row[0].strip())
            sample_id = row[1].strip()

            if not well_pos or not sample_id:
                raise ValueError(
                    f"Fixed map file '{filename}' row {i} contains empty well position or sample ID."
                )

            if well_pos in control_map:
                raise ValueError(f"Well position '{well_pos}' is duplicated in the fixed map file.")

            control_map[well_pos] = sample_id

    # The returned dictionary is passed to the PlateMapper's __init__
    # which will then convert the well strings (e.g., 'A1') into tuple indices (e.g., (0, 0)).
    return control_map


def load_sample_ids(
    filename: str, control_prefix: str | None = None
) -> tuple[list[str], list[str], dict[str, str] | None]:
    """Load sample IDs from a CSV file.

    This function reads a CSV file containing sample IDs. It can also handle
    an optional second column for an initial position map.

    Parameters
    ----------
    filename : str
        Path to the CSV file.
    control_prefix : str, optional
        A prefix used to identify control samples. If a sample ID starts with
        this prefix, it will be treated as a control sample.

    Returns
    -------
    tuple[list[str], list[str], dict[str, str] | None]
        A tuple containing three elements:
        - A list of variable sample IDs.
        - A list of control sample IDs.
        - A dictionary mapping well positions to sample IDs, if an initial
          position map is provided in the CSV file. Otherwise, None.
    """
    all_samples = []
    control_samples = []
    initial_position_map = {}
    has_initial_position_map = False

    with open(filename, "r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        for _, row in enumerate(reader):
            if not row:
                continue

            sample_id = row[0].strip()
            if not sample_id:
                continue

            # Check if the sample row is a control or normal sample
            if control_prefix and sample_id.startswith(control_prefix):
                control_samples.append(sample_id)
            else:
                all_samples.append(sample_id)

            if len(row) > 1 and row[1].strip():
                has_initial_position_map = True
                well_pos = normalize_well_key(row[1].strip())
                # Check if the position is unique in the provided file (can't have 2 samples in 1 position)
                if well_pos in initial_position_map:
                    raise ValueError(
                        f"Well position '{well_pos}' is duplicated in the sample file."
                    )
                initial_position_map[well_pos] = sample_id

    return (
        all_samples,
        control_samples,
        initial_position_map if has_initial_position_map else None,
    )


def save_plate_to_csv(plate_data: np.ndarray, filename: str):
    """Save a single plate map to a CSV file.

    Parameters
    ----------
    plate_data : np.ndarray
        The plate data to save.
    filename : str
        The name of the output CSV file.
    """
    np.savetxt(filename, plate_data, delimiter=",", fmt="%s")
    click.echo(f"Plate map successfully saved to {filename}")


def save_all_plates_to_single_csv(all_plates: list[np.ndarray], filename: str):
    """Save a list of plate maps to a single, combined CSV file.

    Parameters
    ----------
    all_plates : list[np.ndarray]
        A list of plate maps to save.
    filename : str
        The name of the output CSV file.
    """
    with open(filename, "w") as f:
        for i, plate in enumerate(all_plates):
            if i > 0:
                f.write("\n")
            f.write(f"Plate {i + 1}\n")
            np.savetxt(f, plate, delimiter=",", fmt="%s")
    click.echo(f"All {len(all_plates)} plate maps successfully saved to {filename}")
