"""CLI Script for interfacing with code features."""

import os
import random

import click

from wellshuffled.plate_generator import (
    PlateMapperNeighborAware,
    PlateMapperSimple,
    load_sample_ids,
    save_all_plates_to_single_csv,
    save_plate_to_csv,
)


def parse_fixed_map(ctx, param, value):
    """Parse the --fixed-map string into a dictionary of {WELL: SAMPLE_ID}."""
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


@click.command()
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
def main(
    sample_file, output_path, plates, size, simple, separate_files, seed, control_prefix, fixed_map
):
    """
    Generate randomized plate maps from a list of SAMPLE_IDs.

    SAMPLE_FILE: A text file with one sample ID per line.
    OUTPUT_PATH: The path for the output CSV file or directory (if using --separate-files).
    """
    if seed is not None:
        random.seed(seed)
        click.echo(f"Using random seed: {seed}")

    click.echo("--- Plate Map Generator ---")

    # Convert size to integer
    plate_size = int(size)

    # Load samples
    samples, control_samples = load_sample_ids(sample_file, control_prefix=control_prefix)

    total_samples = len(samples) + len(control_samples)
    click.echo(f"Loaded {total_samples} total samples from {os.path.basename(sample_file)}.")
    click.echo(f"  - {len(samples)} variable samples to randomize.")
    if control_prefix:
        click.echo(
            f"  - {len(control_samples)} control samples with fixed positions (Prefix: '{control_prefix}')."
        )

    if fixed_map:
        click.echo(
            "Using MANUALLY DEFINED control map. Skipping Plate 1 randomization for controls."
        )

    # Choose the correct mapper class
    if simple:
        click.echo("Using simple randomization logic.")
        mapper = PlateMapperSimple(
            samples, control_samples, plate_size=plate_size, predefined_control_map=fixed_map
        )
    else:
        click.echo("Using neighbor-aware randomization logic.")
        mapper = PlateMapperNeighborAware(
            samples, control_samples, plate_size=plate_size, predefined_control_map=fixed_map
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


if __name__ == "__main__":
    main()
