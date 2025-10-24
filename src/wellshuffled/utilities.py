"""Utility scripts for `wellshuffled`."""

import csv

import numpy as np


def well_to_index(well: str, plate_dims: tuple[int, int]) -> tuple[int, int]:
    """Convert a standard well designation (e.g., 'A1', 'H12') to (row_index, col_index)."""
    rows, cols = plate_dims

    # Well must be at least two characters (Row letter + Col number)
    if len(well) < 2:
        raise ValueError(f"Invalid well designation: {well}")

    # Determine row index
    row_letter = well[0].upper()
    row_index = ord(row_letter) - ord("A")

    # Determine column index
    try:
        col_number = int(well[1:])
        col_index = col_number - 1
    except ValueError as e:
        raise ValueError(f"Invalid column number in well designation: {well}") from e

    # Validation
    if not (0 <= row_index < rows and 0 <= col_index < cols):
        max_row_letter = chr(ord("A") + rows - 1)
        raise ValueError(
            f"Well {well} is outside plate dimensions ({rows}x{cols}). Max well is {max_row_letter}."
        )

    return row_index, col_index


def load_control_map_from_csv(filename: str) -> dict[str, str]:
    """
    Load a control map from a two-column CSV file (Well, Sample ID).

    :param filename: Path to the CSV file.
    :return: A dictionary mapping well position string (e.g., 'A1') to sample ID.
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

            well_pos = row[0].strip().upper()
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
) -> tuple[list[str], list[str]]:
    """Load in a list of file names from csv file."""
    all_samples = []
    control_samples = []

    with open(filename, "r", newline="") as csv_file:
        for row in csv_file:
            sample_id = row.strip()
            if not sample_id:
                continue

            if control_prefix and sample_id.startswith(control_prefix):
                control_samples.append(sample_id)
            else:
                all_samples.append(sample_id)

    return all_samples, control_samples


def save_plate_to_csv(plate_data: np.ndarray, filename: str):
    """Save a single plate map to a CSV file."""
    np.savetxt(filename, plate_data, delimiter=",", fmt="%s")
    print(f"Plate map successfully saved to {filename}")


def save_all_plates_to_single_csv(all_plates: list[np.ndarray], filename: str):
    """Save a list of plate maps to a single, combined CSV file."""
    with open(filename, "w") as f:
        for i, plate in enumerate(all_plates):
            if i > 0:
                f.write("\n")
            f.write(f"Plate {i + 1}\n")
            np.savetxt(f, plate, delimiter=",", fmt="%s")
    print(f"All {len(all_plates)} plate maps successfully saved to {filename}")
