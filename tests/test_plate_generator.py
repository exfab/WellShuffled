"""Pytest suite for plate_generator.py."""

import random

import numpy as np
import pytest

# Assuming plate_generator.py is in the same directory or accessible via path
from wellshuffled.plate_generator import (
    PlateMapperNeighborAware,
    PlateMapperSimple,
    load_sample_ids,
    save_all_plates_to_single_csv,
)

# --- Fixtures and Setup ---

# Define a standard set of samples for testing
SAMPLES = [f"sample-{i}" for i in range(1, 85)]  # 84 variable samples
CONTROLS = [f"control-{i}" for i in range(85, 97)]  # 12 control samples
ALL_SAMPLES = SAMPLES + CONTROLS
CONTROL_PREFIX = "control-"


# Use a fixed seed for reproducible tests
@pytest.fixture(autouse=True)
def set_random_seed():
    """Set a random seed."""
    random.seed(42)
    np.random.seed(42)


@pytest.fixture
def mock_sample_file(tmp_path):
    """Create a temporary sample file with variable samples and controls."""
    sample_data = "\n".join(ALL_SAMPLES)
    d = tmp_path / "data"
    d.mkdir()
    p = d / "samples.txt"
    p.write_text(sample_data)
    return p


# --- Test load_sample_ids ---


def test_load_sample_ids_with_controls(mock_sample_file):
    """Test loading and separating samples when a control prefix is provided."""
    variable, controls = load_sample_ids(mock_sample_file, control_prefix=CONTROL_PREFIX)

    assert len(variable) == 84
    assert len(controls) == 12
    assert all(s.startswith(CONTROL_PREFIX) for s in controls)
    assert not any(s.startswith(CONTROL_PREFIX) for s in variable)
    assert "sample-1" in variable
    assert "control-85" in controls


def test_load_sample_ids_without_controls(mock_sample_file):
    """Test loading when no control prefix is provided (all samples are variable)."""
    variable, controls = load_sample_ids(mock_sample_file, control_prefix=None)

    assert len(variable) == 96
    assert len(controls) == 0
    assert "control-85" in variable


# --- Utility Functions for Plate Validation ---


def validate_plate_contents(plate, expected_total_samples):
    """Check plate dimensions and uniqueness of samples."""
    assert plate.size == 96

    unique_samples = set(plate.flatten().tolist())
    # Should contain all 96 samples (84 variable + 12 control)
    assert len(unique_samples) == expected_total_samples
    assert unique_samples == set(ALL_SAMPLES)


def get_control_positions(plate, controls):
    """Return a dictionary mapping (row, col) position to control ID.

    Matches the format of mapper.fixed_control_map: {(R, C): Control ID}.
    """
    pos_map = {}
    rows, cols = plate.shape
    for r in range(rows):
        for c in range(cols):
            sample = plate[r, c]
            if sample in controls:
                pos_map[(r, c)] = sample
    return pos_map


# --- PlateMapperSimple Tests ---


def test_simple_mapper_plate_generation_basic():
    """Test basic functionality of the Simple Mapper."""
    mapper = PlateMapperSimple(SAMPLES, CONTROLS, plate_size=96)
    plate = mapper.generate_plate()

    # Check dimensions
    assert plate.shape == (8, 12)

    # Check contents
    validate_plate_contents(plate, 96)

    # Check edge tracking: The number of variable samples used on the edge
    # must be > 0 and <= 40 (total edge wells).
    assert 0 < len(mapper.used_edge_samples) <= 40
    assert not mapper.multi_edge_samples


def test_simple_mapper_control_fixing():
    """Test that control positions are randomized on Plate 1 and fixed on Plate 2."""
    mapper = PlateMapperSimple(SAMPLES, CONTROLS, plate_size=96)

    # Plate 1 (Initial randomization)
    plate1 = mapper.generate_plate()
    pos_map1 = get_control_positions(plate1, CONTROLS)

    # Check that controls are fixed after plate 1
    assert mapper.is_control_map_fixed
    assert len(mapper.fixed_control_map) == len(CONTROLS)
    assert pos_map1 == mapper.fixed_control_map  # Now comparing same structure

    # Plate 2 (Fixed positions)
    plate2 = mapper.generate_plate()
    pos_map2 = get_control_positions(plate2, CONTROLS)

    # Check that positions are identical
    assert pos_map1 == pos_map2

    # Check that variable samples are different
    assert not np.array_equal(plate1, plate2)


# --- PlateMapperNeighborAware Tests ---


def test_neighbor_aware_mapper_control_fixing():
    """Test that NeighborAware Mapper correctly fixes control positions (the critical fix)."""
    mapper = PlateMapperNeighborAware(SAMPLES, CONTROLS, plate_size=96)

    # Plate 1 (Initial randomization)
    plate1 = mapper.generate_plate()
    pos_map1 = get_control_positions(plate1, CONTROLS)

    # Check that controls are fixed after plate 1
    assert mapper.is_control_map_fixed
    assert pos_map1 == mapper.fixed_control_map  # Now comparing same structure

    # Plate 2 (Fixed positions)
    plate2 = mapper.generate_plate()
    pos_map2 = get_control_positions(plate2, CONTROLS)

    # CRITICAL CHECK: Positions must be identical
    assert pos_map1 == pos_map2

    # Check that variable samples are different
    assert not np.array_equal(plate1, plate2)


def test_neighbor_aware_mapper_neighbor_tracking():
    """Test that neighbor pairs are tracked correctly after generation."""
    mapper = PlateMapperNeighborAware(SAMPLES, CONTROLS, plate_size=96)
    plate1 = mapper.generate_plate()

    # Check that neighbor pairs were generated
    assert len(mapper.neighbor_pairs) > 0

    # Total unique horizontal (8*11) + vertical (12*7) pairs = 172
    # Since all wells are filled, the number of unique pairs should be EXACTLY 172.
    assert len(mapper.neighbor_pairs) == 172

    # Check a specific pair constraint (hard to check avoidance, but we can verify the set contents)
    # Get the sample at (0, 0) and (0, 1)
    sample_a = plate1[0, 0]
    sample_b = plate1[0, 1]

    pair = tuple(sorted((sample_a, sample_b)))
    assert pair in mapper.neighbor_pairs


# --- Test File Output ---


def test_save_all_plates_to_single_csv(tmp_path):
    """Test saving multiple plates to a single file."""
    mapper = PlateMapperSimple(SAMPLES, CONTROLS, plate_size=96)
    plates = mapper.generate_multiple_plates(num_plates=3)

    output_path = tmp_path / "combined_output.csv"
    save_all_plates_to_single_csv(plates, output_path)

    # Check file content structure
    content = output_path.read_text()
    assert "Plate 1" in content
    assert "Plate 2" in content
    assert "Plate 3" in content
    assert content.count("\n") > (8 * 3)  # Should have at least 8 rows * 3 plates + separators
    assert content.count(f",{CONTROLS[0]},") > 0  # Check for control sample separator presence
