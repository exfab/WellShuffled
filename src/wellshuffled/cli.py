"""CLI Script for interfacing with code features."""

import csv
import os
import random

import click

from wellshuffled.plate_generator import PlateMapperNeighborAware, PlateMapperSimple
from wellshuffled.utilities import (
    convert_position_to_well_number,
    load_control_map_from_csv,
    load_sample_ids,
    save_all_plates_to_single_csv,
    save_plate_to_csv,
)


def parse_fixed_map(
    ctx: click.Context, param: click.Parameter, value: str
) -> dict[str, str] | None:
    """Parse the --fixed-map string into a dictionary of {WELL: SAMPLE_ID}.

    Parameters
    ----------
    ctx : click.Context
        The click context.
    param : click.Parameter
        The click parameter.
    value : str
        The value of the --fixed-map option.

    Returns
    -------
    dict[str, str] | None
        A dictionary mapping well positions to sample IDs, or None if the
        value is empty.

    Raises
    ------
    click.BadParameter
        If the --fixed-map string is malformed.
    """
    if not value:
        return None

    fixed_map = {}
    try:
        # Expected format: "A1:control-85,H12:control-96"
        pairs = value.split(",")
        if not pairs:
            raise ValueError("Map is empty.")

        for pair in pairs:
            if ":" not in pair:
                raise ValueError(f"Each assignment must be in WELL:SAMPLE_ID format. Got '{pair}'.")

            # Split into well and sample_id, using maxsplit=1 in case SAMPLE_ID contains a colon
            parts = pair.strip().split(":", 1)
            if len(parts) != 2:
                raise ValueError(f"Each assignment must be in WELL:SAMPLE_ID format. Got '{pair}'.")

            well, sample_id = parts
            fixed_map[well.upper()] = sample_id.strip()

        return fixed_map
    except Exception as e:
        # Re-raise as a BadParameter for Click to handle gracefully
        raise click.BadParameter(f"Could not parse --fixed-map string: {e}") from None


def parse_fixed_map_file(
    ctx: click.Context, param: click.Parameter, value: str
) -> dict[str, str] | None:
    """Parse the --fixed-map-file path and load data.

    Parameters
    ----------
    ctx : click.Context
        The click context.
    param : click.Parameter
        The click parameter.
    value : str
        The value of the --fixed-map-file option.

    Returns
    -------
    dict[str, str] | None
        A dictionary mapping well positions to sample IDs, or None if the
        value is empty.

    Raises
    ------
    click.BadParameter
        If the file is not found or cannot be parsed.
    """
    if not value:
        return None

    # We pass the file path to the utility function in plate_generator
    try:
        fixed_map = load_control_map_from_csv(value)
        if not fixed_map:
            raise click.BadParameter("Fixed map file is empty or contains no valid data.")
        return fixed_map
    except Exception as e:
        # Catch any exceptions during file reading/parsing and report to the user
        raise click.BadParameter(f"Error reading fixed map file '{value}': {e}") from e


def parse_dimensions(
    ctx: click.Context, param: click.Parameter, value: str
) -> tuple[int, int] | None:
    """Parse a string like '8,12' or '8x12' into a tuple (rows, cols).

    Parameters
    ----------
    ctx : click.Context
        The click context.
    param : click.Parameter
        The click parameter.
    value : str
        The value of the --nonstandard_dims option.

    Returns
    -------
    tuple[int, int] | None
        A tuple containing the number of rows and columns, or None if the
        value is empty.

    Raises
    ------
    click.BadParameter
        If the dimensions string is malformed.
    """
    if not value:
        return None

    try:
        # distinct separator handling
        if "," in value:
            parts = value.split(",")
        elif "x" in value.lower():
            parts = value.lower().split("x")
        else:
            # Fallback for space separated if passed as a single quoted string
            parts = value.split()

        if len(parts) != 2:
            raise ValueError("Dimensions must be two numbers (rows, cols).")

        return (int(parts[0]), int(parts[1]))
    except Exception as e:
        raise click.BadParameter(
            f"Dimensions must be in format 'rows,cols' (e.g., 8,12). Got: {value}"
        ) from e


@click.group()
def wellshuffled():
    """CLI for generating randomized plate maps."""
    click.echo("WellShuffled: A tool for generating randomized plate maps.")


@wellshuffled.command()
@click.argument("sample_file", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.argument("output_path", type=click.Path())
@click.option(
    "--plates", "-n", default=1, type=int, show_default=True, help="Number of plates to generate."
)
@click.option(
    "--size",
    default=96,
    type=click.Choice(["96", "384"]),
    show_default=True,
    help="Well plate size (96 or 384).",
)
@click.option(
    "--simple", is_flag=True, help="Use simple randomization (disables neighbor-awareness)."
)
@click.option(
    "--separate-files",
    is_flag=True,
    help="Save each plate map to a separate CSV file in a directory.",
)
@click.option(
    "--seed", type=int, default=None, help="Set the random seed for reproducible results."
)
@click.option(
    "--control-prefix",
    default=None,
    help="Prefix used to identify control/blank samples in SAMPLE_FILE (e.g., 'B', 'CTRL').",
)
@click.option(
    "--fixed-map",
    default=None,
    callback=parse_fixed_map,
    help="Manually specify fixed control locations (e.g., 'A1:control-85,H12:control-96'). Overrides Plate 1 randomization.",
)
@click.option(
    "--fixed-map-file",
    default=None,
    callback=parse_fixed_map_file,
    help="Manually specify fixed control locations from a csv file (e.g well_pos, sample_id). Overrides Plate 1 randomization.",
)
@click.option(
    "--nonstandard",
    is_flag=True,
    default=False,
    help="A flag to allow for the use of non-standard plate dimensions (not 48, 96, 384, etc)",
)
@click.option(
    "--nonstandard_dims",
    default=None,
    callback=parse_dimensions,
    help="The dimensions (x, y) for the nonstandard plate",
)
def shuffle(
    sample_file: str,
    output_path: str,
    plates: int,
    size: str,
    simple: bool,
    separate_files: bool,
    seed: int | None,
    control_prefix: str | None,
    fixed_map: dict[str, str] | None,
    fixed_map_file: dict[str, str] | None,
    nonstandard: bool,
    nonstandard_dims: tuple[int, int] | None,
) -> None:
    """Generate randomized plate maps from a list of sample IDs.

    Parameters
    ----------
    sample_file : str
        Path to a text file with one sample ID per line.
    output_path : str
        Path for the output CSV file or directory.
    plates : int
        Number of plates to generate.
    size : str
        Well plate size (96 or 384).
    simple : bool
        Use simple randomization (disables neighbor-awareness).
    separate_files : bool
        Save each plate map to a separate CSV file in a directory.
    seed : int, optional
        Set the random seed for reproducible results.
    control_prefix : str, optional
        Prefix used to identify control/blank samples in the sample file.
    fixed_map : dict[str, str], optional
        Manually specify fixed control locations.
    fixed_map_file : str, optional
        Path to a CSV file with fixed control locations.
    nonstandard : bool
        Allow for the use of non-standard plate dimensions.
    nonstandard_dims : tuple[int, int], optional
        The dimensions (rows, cols) for the nonstandard plate.
    """
    if seed is not None:
        random.seed(seed)
        click.echo(f"Using random seed: {seed}")

    click.echo("--- Plate Map Generator ---")

    # Convert size to integer
    if nonstandard and nonstandard_dims:
        plate_size = int(nonstandard_dims[0] * nonstandard_dims[1])
    else:
        plate_size = int(size)

    # Load samples
    samples, control_samples, initial_position_map = load_sample_ids(
        sample_file, control_prefix=control_prefix
    )

    total_samples = len(samples) + len(control_samples)
    click.echo(f"Loaded {total_samples} total samples from {os.path.basename(sample_file)}.")
    click.echo(f"  - {len(samples)} variable samples to randomize.")
    if control_prefix:
        click.echo(
            f"  - {len(control_samples)} control samples with fixed positions (Prefix: '{control_prefix}')."
        )

    # Log that we are using the input well positions for the first plate, don't need to do anything else.
    if initial_position_map:
        click.echo("Sample file contains initial plate positions, using it for Plate 1.")

    # Identify if we have a fixed control map (either as input text or a file.)
    if fixed_map or fixed_map_file:
        click.echo(
            "Using MANUALLY DEFINED control map. Skipping Plate 1 randomization for controls."
        )
        if fixed_map_file:
            fixed_map = fixed_map_file

    # Choose the correct mapper class
    mapper: PlateMapperSimple | PlateMapperNeighborAware
    if simple:
        click.echo("Using simple randomization logic, minimizing repeated samples on the edge.")
        mapper = PlateMapperSimple(
            samples,
            control_samples,
            plate_size=plate_size,
            predefined_control_map=fixed_map,
            nonstandard=nonstandard,
            nonstandard_dims=nonstandard_dims,
            initial_position_map=initial_position_map,
        )
    else:
        click.echo(
            "Using neighbor-aware randomization logic, minimizing repeated samples on the edge and repeated sample neighbors."
        )
        mapper = PlateMapperNeighborAware(
            samples,
            control_samples,
            plate_size=plate_size,
            predefined_control_map=fixed_map,
            nonstandard=nonstandard,
            nonstandard_dims=nonstandard_dims,
            initial_position_map=initial_position_map,
        )

    # Generate plates
    click.echo(f"Generating {plates} plate(s) of size {plate_size}...")
    all_plates = mapper.generate_multiple_plates(num_plates=plates)

    # Save output
    if separate_files:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for i, plate in enumerate(all_plates):
            filename = os.path.join(output_path, f"plate_map_{i + 1}.csv")
            save_plate_to_csv(plate, filename)
    else:
        save_all_plates_to_single_csv(all_plates, output_path)

    # Final report
    click.echo("\n--- Summary ---")
    click.echo(f"Total unique samples used on an edge: {len(mapper.used_edge_samples)}")
    if mapper.multi_edge_samples:
        click.echo("Samples re-used on edges (by plate):")
        click.echo(str(mapper.multi_edge_samples))


def _process_plate_data(
    plate_data: list[list[str]],
    plate_index: int,
    trajectories: dict[str, list[str]],
    use_numeric_wells: bool | None = False,
) -> None:
    """Parse a single plate's rows and update trajectories.

    Parameters
    ----------
    plate_data : list[list[str]]
        A 2D list representing the plate data.
    plate_index : int
        The index of the plate being processed.
    trajectories : dict[str, list[str]]
        A dictionary to store the sample trajectories.
    use_numeric_wells : bool, optional
        A flag to indicate whether to use numeric well positions.
    """
    # Pull dimensions of plate
    try:
        num_rows = len(plate_data)
        if num_rows == 0:
            return

        num_cols = len(plate_data[0])
        if num_cols == 0:
            return

        plate_dims = (num_rows, num_cols)
    except IndexError:
        click.echo(f"Warning: Plate {plate_index} appears to be empty or malformed")
        return

    for r, row_data in enumerate(plate_data):
        for c, sample_id in enumerate(row_data):
            if sample_id and sample_id != "None":
                # Convert 0-indexed (r, c) to standard well position (e.g., (0, 0) -> A1)
                row_letter = chr(ord("A") + r)
                col_number = c + 1
                well_pos_alpha = f"{row_letter}{col_number}"

                if use_numeric_wells:
                    position_to_append = convert_position_to_well_number(well_pos_alpha, plate_dims)
                else:
                    position_to_append = well_pos_alpha

                if sample_id not in trajectories:
                    trajectories[sample_id] = []

                trajectories[sample_id].append(position_to_append)


@wellshuffled.command()
@click.argument("input_path", type=click.Path(exists=True, resolve_path=True))
@click.option(
    "--output-csv",
    type=click.Path(),
    default=None,
    help="Optional path to save the full trajectory map as a CSV file.",
)
@click.option(
    "--numeric",
    "use_numeric_wells",
    is_flag=True,
    default=False,
    help="Return the plate positions as 1-based column major numeric values.",
)
def trace(input_path: str, output_csv: str, use_numeric_wells: bool = False) -> None:
    """Trace the samples over their various plates.

    Parameters
    ----------
    input_path : str
        Path to a directory of plate files or a single combined plate file.
    output_csv : str, optional
        Path to save the full trajectory map as a CSV file.
    use_numeric_wells : bool, optional
        Return the plate positions as 1-based column major numeric values.
    """
    trajectories: dict[str, list[str]] = {}

    # 1. Determine files to process
    if os.path.isdir(input_path):
        # Scenario: Directory of separate files
        file_paths = sorted([
            os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".csv")
        ])
        if not file_paths:
            click.echo(f"Error: No CSV files found in directory: {input_path}")
            return

        click.echo(f"Processing {len(file_paths)} separate plate files from {input_path}...")

        # Process separate files sequentially
        for i, file_path in enumerate(file_paths, start=1):
            try:
                with open(file_path, "r", newline="") as f:
                    reader = csv.reader(f)
                    plate_data = list(reader)
                    _process_plate_data(plate_data, i, trajectories, use_numeric_wells)
            except Exception as e:
                click.echo(f"Error reading plate file {file_path}: {e}")
                return

    else:
        # Scenario: Single combined file
        click.echo(f"Processing combined plate map: {input_path}...")
        current_plate_index = 0
        plate_data = []

        try:
            with open(input_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith("Plate "):
                        # Process previous plate if it exists
                        if plate_data:
                            _process_plate_data(
                                plate_data, current_plate_index, trajectories, use_numeric_wells
                            )

                        # Start of new plate
                        try:
                            # Extract plate number from "Plate X" header
                            current_plate_index = int(line.split(" ")[1])
                        except (IndexError, ValueError):
                            # Default to sequential indexing if format is unexpected
                            current_plate_index += 1
                        plate_data = []  # Reset for new plate
                    else:
                        # Collect well data from CSV line
                        row_data = [cell.strip() for cell in line.split(",")]
                        plate_data.append(row_data)

                # Process the last plate block
                if plate_data:
                    _process_plate_data(
                        plate_data, current_plate_index, trajectories, use_numeric_wells
                    )

        except Exception as e:
            click.echo(f"An error occurred while reading the combined CSV: {e}")
            return

    # Output the data
    if not trajectories:
        click.echo("Error: No sample trajectories found in the provided path(s).")
        return

    click.echo("\n--- Sample Trajectories Across Plates ---")

    # Sort samples alphabetically for predictable output
    sorted_samples = sorted(trajectories.keys())

    num_plates = max(len(path) for path in trajectories.values()) if trajectories else 0

    for sample_id in sorted_samples:
        path = " -> ".join(trajectories[sample_id])
        click.echo(f"{sample_id}: {path}")

    # Save results to CSV file
    if output_csv:
        try:
            with open(output_csv, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                header = ["Sample_ID"] + [f"Plate {i + 1}" for i in range(num_plates)]
                writer.writerow(header)

                for sample_id in sorted_samples:
                    row = [sample_id] + trajectories[sample_id]
                    writer.writerow(row)

            click.echo(f"\nSuccessfully saved all trajectories to: {output_csv}")
        except Exception as e:
            click.echo(f"\nError saving trajectory CSV to {output_csv}: {e}")


if __name__ == "__main__":
    wellshuffled()
