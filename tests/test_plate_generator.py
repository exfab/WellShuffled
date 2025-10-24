"""Pytest suite for the plate_generator module."""

import random

import numpy as np
import pytest

from wellshuffled.plate_generator import PlateMapperNeighborAware, PlateMapperSimple
from wellshuffled.utilities import load_sample_ids, well_to_index

# --- Test Data ---

# Create mock data (70 variable samples, 12 controls for a 96-well plate)
SAMPLES = [f"sample-{i + 1}" for i in range(70)]
CONTROLS = [f"control-{i + 85}" for i in range(12)]
ALL_SAMPLES = SAMPLES + CONTROLS
SAMPLE_FILE_CONTENT = "\n".join(ALL_SAMPLES)
TOTAL_SAMPLES = len(ALL_SAMPLES)  # 82

# Define a complete manual map for testing the new feature (all 12 controls)
manual_wells = [f"A{i + 1}" for i in range(12)]
MANUAL_CONTROL_MAP = dict(zip(manual_wells, CONTROLS, strict=True))

# Expected internal map for a 96-well plate (8x12)
EXPECTED_FIXED_MAP_96 = {(0, i): CONTROLS[i] for i in range(12)}

# --- Fixtures and Helpers ---


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set a fixed seed for reproducible test results."""
    random.seed(42)
    np.random.seed(42)


@pytest.fixture
def sample_file_path(tmp_path):
    """Create a temporary sample file for testing load_sample_ids."""
    p = tmp_path / "samples.txt"
    p.write_text(SAMPLE_FILE_CONTENT)
    return str(p)


def validate_plate_contents(plate: np.ndarray, expected_filled_count: int):
    """Help to check shape and content uniqueness."""
    # Check that the plate is filled with the expected number of samples (82, not 96)
    assert np.count_nonzero(plate != None) == expected_filled_count  # noqa: E711

    # Check that all placed samples are unique on the plate
    flat_plate = [s for s in plate.flatten().tolist() if s is not None]
    assert len(flat_plate) == len(set(flat_plate))


def get_control_positions(plate: np.ndarray, control_ids: list[str]) -> dict[tuple, str]:
    """Help to extract control sample positions in the (R, C): Sample ID format."""
    positions = {}
    rows, cols = plate.shape
    for r in range(rows):
        for c in range(cols):
            sample = plate[r, c]
            if sample in control_ids:
                positions[(r, c)] = sample
    return positions


# --- Test Cases ---


def test_load_sample_ids(sample_file_path):
    """Test that sample loading correctly separates variable and control samples."""
    # FIX: Use tmp_path fixture for file access, resolving the TypeError
    variable, control = load_sample_ids(sample_file_path, control_prefix="control-")
    assert len(variable) == 70
    assert len(control) == 12
    assert "sample-1" in variable
    assert "control-85" in control


def test_well_to_index():
    """Test the coordinate conversion utility."""
    assert well_to_index("A1", (8, 12)) == (0, 0)
    assert well_to_index("h12", (8, 12)) == (7, 11)
    assert well_to_index("B5", (8, 12)) == (1, 4)
    assert well_to_index("P24", (16, 24)) == (15, 23)

    with pytest.raises(ValueError):
        well_to_index("A13", (8, 12))  # Out of bounds column
    with pytest.raises(ValueError):
        well_to_index("I1", (8, 12))  # Out of bounds row


def test_simple_mapper_plate_generation_basic():
    """Test basic dimensions and edge tracking for the Simple Mapper."""
    mapper = PlateMapperSimple(SAMPLES, CONTROLS, plate_size=96)
    plate = mapper.generate_plate()

    # Check dimensions (8x12 for 96-well)
    assert plate.shape == (8, 12)

    # FIX: Check contents against the total number of samples (82), resolving AssertionError
    validate_plate_contents(plate, TOTAL_SAMPLES)

    # Check edge tracking (should be 40 total edge spots, up to 12 of which are controls)
    # The number of *variable* samples used on the edge must be in a reasonable range.
    assert 0 < len(mapper.used_edge_samples) <= 40


def test_simple_mapper_control_fixing():
    """Test that control positions are randomized on Plate 1 and fixed on Plate 2."""
    mapper = PlateMapperSimple(SAMPLES, CONTROLS, plate_size=96)

    plate1 = mapper.generate_plate()
    pos_map1 = get_control_positions(plate1, CONTROLS)

    assert mapper.is_control_map_fixed

    # The control positions on Plate 1 must be the fixed map
    assert pos_map1 == mapper.fixed_control_map

    plate2 = mapper.generate_plate()
    pos_map2 = get_control_positions(plate2, CONTROLS)

    # Positions must be identical across both plates
    assert pos_map1 == pos_map2

    assert not np.array_equal(plate1, plate2)


def test_mapper_manual_control_fixing():
    """Test that manually defined control positions are used for all plates."""
    # FIX: MANUAL_CONTROL_MAP now contains all 12 controls, resolving ValueError.

    # Simple Mapper Test
    mapper_s = PlateMapperSimple(
        SAMPLES, CONTROLS, plate_size=96, predefined_control_map=MANUAL_CONTROL_MAP
    )

    assert mapper_s.is_control_map_fixed
    assert mapper_s.fixed_control_map == EXPECTED_FIXED_MAP_96

    plate1_s = mapper_s.generate_plate()
    plate2_s = mapper_s.generate_plate()

    assert get_control_positions(plate1_s, CONTROLS) == EXPECTED_FIXED_MAP_96
    assert get_control_positions(plate2_s, CONTROLS) == EXPECTED_FIXED_MAP_96

    # Neighbor-Aware Mapper Test
    mapper_n = PlateMapperNeighborAware(
        SAMPLES, CONTROLS, plate_size=96, predefined_control_map=MANUAL_CONTROL_MAP
    )

    assert mapper_n.is_control_map_fixed
    assert mapper_n.fixed_control_map == EXPECTED_FIXED_MAP_96

    plate1_n = mapper_n.generate_plate()
    plate2_n = mapper_n.generate_plate()

    assert get_control_positions(plate1_n, CONTROLS) == EXPECTED_FIXED_MAP_96
    assert get_control_positions(plate2_n, CONTROLS) == EXPECTED_FIXED_MAP_96


def test_neighbor_aware_mapper_control_fixing():
    """Test that NeighborAware Mapper correctly fixes control positions."""
    mapper = PlateMapperNeighborAware(SAMPLES, CONTROLS, plate_size=96)

    plate1 = mapper.generate_plate()
    pos_map1 = get_control_positions(plate1, CONTROLS)

    assert mapper.is_control_map_fixed
    assert pos_map1 == mapper.fixed_control_map

    plate2 = mapper.generate_plate()
    pos_map2 = get_control_positions(plate2, CONTROLS)

    assert pos_map1 == pos_map2
    assert not np.array_equal(plate1, plate2)


def test_neighbor_aware_mapper_neighbor_tracking():
    """Test that neighbor pairs are tracked correctly after generation."""
    mapper = PlateMapperNeighborAware(SAMPLES, CONTROLS, plate_size=96)
    mapper.generate_plate()

    # FIX: The expected number of pairs is LESS THAN 172 since the plate is not full.
    # The actual number will vary slightly, but must be > 0 and < 172.
    assert 0 < len(mapper.neighbor_pairs) < 172

    mapper.generate_plate()

    # After plate 2, the number of unique neighbor pairs should have increased.
    plate1_pairs = mapper.neighbor_pairs.copy()

    mapper_p2 = PlateMapperNeighborAware(SAMPLES, CONTROLS, plate_size=96)
    mapper_p2.neighbor_pairs = plate1_pairs
    mapper_p2.generate_plate()

    # The number of unique pairs should be significantly higher than 172
    assert len(mapper_p2.neighbor_pairs) > 172
