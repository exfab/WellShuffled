"""Script for generating randomized plates for various configurations."""

import random
from abc import ABC, abstractmethod

import click
import numpy as np

from wellshuffled.utilities import (
    well_to_index,
)


class BasePlateMapper(ABC):
    """Base class for shared PlateMapper functionality.

    Parameters
    ----------
    sample_ids : list[str]
        A list of sample IDs to be placed on the plate.
    control_sample_ids : list[str]
        A list of control sample IDs to be placed on the plate.
    plate_size : int, optional
        The size of the plate (96 or 384). Default is 96.
    predefined_control_map : dict[str, str], optional
        A dictionary mapping well positions to control sample IDs. These
        positions will be fixed across all generated plates.
    nonstandard : bool, optional
        A flag to allow for the use of non-standard plate dimensions.
    nonstandard_dims : tuple[int, int], optional
        The dimensions (rows, cols) for the nonstandard plate.
    initial_position_map : dict[str, str], optional
        A dictionary mapping well positions to sample IDs for the first plate.

    Attributes
    ----------
    samples : list[str]
        A list of variable sample IDs.
    control_samples : list[str]
        A list of control sample IDs.
    all_samples : list[str]
        A list of all sample IDs.
    initial_position_map : dict[str, str] | None
        A dictionary mapping well positions to sample IDs for the first plate.
    plate_dims : tuple[int, int]
        The dimensions of the plate (rows, cols).
    plate_size : int
        The total number of wells on the plate.
    fixed_control_map : dict[tuple[int, int], str]
        A dictionary mapping (row, col) tuples to control sample IDs.
    is_control_map_fixed : bool
        A flag indicating whether the control map is fixed.
    partial_plate : bool
        A flag indicating whether the plate is partially filled.
    used_edge_samples : set[str]
        A set of samples that have been used on the edge of a plate.
    multi_edge_samples : list[list[str]]
        A list of lists of samples that have been used on the edge of a plate
        more than once.
    """

    def __init__(
        self,
        sample_ids: list[str],
        control_sample_ids: list[str],
        plate_size: int = 96,
        predefined_control_map: dict[str, str] | None = None,
        nonstandard=False,
        nonstandard_dims: tuple[int, int] | None = None,
        initial_position_map: dict[str, str] | None = None,
    ):
        # Check plate sizes and presence of nonstandard dimensions
        if plate_size not in [96, 384] and not nonstandard:
            raise ValueError("Plate size unknown, must be 96 or 384.")

        if nonstandard and not nonstandard_dims:
            raise ValueError("Non-standard mode requires `nonstandard_dims` tuple.")

        # Parse the samples list into samples and controls, if we pass in source plate positions, track those.
        self.samples = list(sample_ids)
        self.control_samples = list(control_sample_ids)
        self.all_samples = self.samples + self.control_samples

        # Check if we pass in the initial plate positions
        self.initial_position_map = (
            initial_position_map if initial_position_map is not None else None
        )

        # We check if the plate is standard or non-standard and set the plate size and dimensions based on that
        if nonstandard and nonstandard_dims:
            self.plate_dims = nonstandard_dims
            self.plate_size = nonstandard_dims[0] * nonstandard_dims[1]
        elif nonstandard and not nonstandard_dims:
            raise ValueError("Non-standard flag raised, but no nonstandard_dims provided!")
        else:
            self.plate_size = plate_size
            self.plate_dims = (16, 24) if plate_size == 384 else (8, 12)

        # Create our fixed control map
        self.fixed_control_map: dict[tuple[int, int], str] = {}
        self.is_control_map_fixed = False

        # Check the rows and cols of the plate we have as input
        rows, cols = self.plate_dims
        all_indices = rows * cols

        # Check for filling a partial plate
        if len(self.all_samples) > all_indices:
            raise ValueError(
                f"Total Samples ({len(self.all_samples)}) exceeds wells ({all_indices})."
            )
        elif len(self.all_samples) < all_indices:
            click.echo(f"Fitting {len(self.all_samples)} samples into plate of size {plate_size}.")
            self.partial_plate = True
        else:
            self.partial_plate = False

        # Track state of edge samples, and sample neighbors
        self.used_edge_samples: set[str] = set()
        self.multi_edge_samples: list[list[str]] = []

        # If the positions of the controls were predefined, store their positions in the fixed_control_map
        if self.initial_position_map and predefined_control_map:
            # Create a reverse map for the initial position map for easier lookup
            initial_pos_reverse_map = {v: k for k, v in self.initial_position_map.items()}

            for well, sample_id in predefined_control_map.items():
                if (
                    sample_id in initial_pos_reverse_map
                    and initial_pos_reverse_map[sample_id] != well
                ):
                    raise ValueError(
                        f"The position for control sample {sample_id} is different in the initial position map and the fixed control map."
                    )
                if (
                    well in self.initial_position_map
                    and self.initial_position_map[well] != sample_id
                ):
                    raise ValueError(
                        f"The sample ID for well {well} is different in the initial position map and the fixed control map."
                    )

        if predefined_control_map:
            # Manual control map provided
            for well, sample_id in predefined_control_map.items():
                if sample_id not in self.control_samples:
                    raise ValueError(
                        f"Sample '{sample_id}' in the fixed map is not in the control samples list."
                    )
                r, c = well_to_index(well, self.plate_dims)
                self.fixed_control_map[(r, c)] = sample_id

            self.is_control_map_fixed = True

        elif not self.control_samples:
            # If no controls, no map needed.
            self.is_control_map_fixed = True

    @abstractmethod
    def generate_plate(self) -> np.ndarray:
        """Generate a single randomized plate map.

        Returns
        -------
        np.ndarray
            A 2D numpy array representing the plate map.
        """
        pass

    def _generate_initial_position_plate(self) -> np.ndarray:
        """Generate a plate from the initial position map.

        Returns
        -------
        np.ndarray
            A 2D numpy array representing the plate map.
        """
        plate = np.full(self.plate_dims, None, dtype=object)
        if self.initial_position_map:
            merged_map = self.initial_position_map.copy()
        else:
            raise ValueError("No initial_position_map supplied!")

        if self.is_control_map_fixed:
            for (r, c), sample_id in self.fixed_control_map.items():
                # Convert (r, c) back to well position string
                well = f"{chr(ord('A') + r)}{c + 1}"
                merged_map[well] = sample_id

        for well, sample_id in merged_map.items():
            r, c = well_to_index(well, self.plate_dims)
            plate[r, c] = sample_id
        return plate

    def _get_perimeter_indices(self) -> tuple[list[tuple], list[tuple]]:
        """Calculate the (row, col) indices for perimeter and interior well positions.

        Returns
        -------
        tuple[list[tuple], list[tuple]]
            A tuple containing two lists:
            - A list of (row, col) tuples for the perimeter wells.
            - A list of (row, col) tuples for the interior wells.
        """
        rows, cols = self.plate_dims
        perimeter_indices = []
        interior_indices = []

        for r in range(rows):
            for c in range(cols):
                if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                    perimeter_indices.append((r, c))
                else:
                    interior_indices.append((r, c))

        return perimeter_indices, interior_indices

    def generate_multiple_plates(self, num_plates: int) -> list[np.ndarray]:
        """Generate a specified number of unique plate layouts.

        Parameters
        ----------
        num_plates : int
            The number of plates to generate.

        Returns
        -------
        list[np.ndarray]
            A list of 2D numpy arrays, each representing a plate map.
        """
        plates = []
        if self.initial_position_map:
            # Generate the first plate from the predefined map
            predefined_plate = self._generate_initial_position_plate()
            plates.append(predefined_plate)
            # Update neighbor state if applicable
            if isinstance(self, PlateMapperNeighborAware):
                self._update_neighbor_state(predefined_plate)
            # Reduce the number of plates to generate randomly
            num_plates -= 1
            self.initial_position_map = None  # Only use it once

        for _ in range(num_plates):
            plates.append(self.generate_plate())

        return plates


class PlateMapperSimple(BasePlateMapper):
    """Generate and manage randomized plate maps without neighbor-awareness."""

    def __init__(
        self,
        sample_ids: list[str],
        control_sample_ids: list[str],
        plate_size: int = 96,
        predefined_control_map: dict[str, str] | None = None,
        nonstandard=False,
        nonstandard_dims: tuple[int, int] | None = None,
        initial_position_map: dict[str, str] | None = None,
    ):
        super().__init__(
            sample_ids,
            control_sample_ids,
            plate_size,
            predefined_control_map,
            nonstandard,
            nonstandard_dims,
            initial_position_map,
        )

    def generate_plate(self) -> np.ndarray:
        """Generate a single randomized plate map.

        Returns
        -------
        np.ndarray
            A 2D numpy array representing the plate map.
        """
        plate = np.full(self.plate_dims, None, dtype=object)

        if self.is_control_map_fixed:
            # Place in the controls first!
            for (r, c), sample in self.fixed_control_map.items():
                plate[r, c] = sample

            available_indices = {
                (r, c) for r in range(self.plate_dims[0]) for c in range(self.plate_dims[1])
            }
            fixed_indices = set(self.fixed_control_map.keys())
            variable_indices = list(available_indices - fixed_indices)

            all_perimeter, all_interior = self._get_perimeter_indices()
            perimeter_indices = [idx for idx in all_perimeter if idx in variable_indices]
            interior_indices = [idx for idx in all_interior if idx in variable_indices]

            samples_to_randomize = list(self.samples)

        else:
            all_indices = [
                (r, c) for r in range(self.plate_dims[0]) for c in range(self.plate_dims[1])
            ]
            random.shuffle(all_indices)

            all_perimeter, all_interior = self._get_perimeter_indices()
            perimeter_indices = [idx for idx in all_perimeter if idx in all_indices]
            interior_indices = [idx for idx in all_interior if idx in all_indices]

            # All samples (variable + control) are candidates for randomization
            samples_to_randomize = list(self.all_samples)

        num_perimeter_wells = len(perimeter_indices)

        # Find samples that have not been used on the edge yet.
        samples_for_edge = [s for s in samples_to_randomize if s in self.samples]

        available_edge_samples = [s for s in samples_for_edge if s not in self.used_edge_samples]
        random.shuffle(available_edge_samples)

        # Find samples that have already been used on an edge.
        recycled_edge_samples = list(self.used_edge_samples)
        random.shuffle(recycled_edge_samples)

        if not self.is_control_map_fixed:
            random.shuffle(samples_to_randomize)

            edge_placements = samples_to_randomize[:num_perimeter_wells]
            samples_to_place = samples_to_randomize[num_perimeter_wells:]

        else:
            # Select samples for the edge
            edge_placements = available_edge_samples[:num_perimeter_wells]

            # If not enough fresh samples, supplement with already-used ones but keep track
            if len(edge_placements) < num_perimeter_wells:
                needed = num_perimeter_wells - len(edge_placements)
                edge_placements.extend(recycled_edge_samples[:needed])
                self.multi_edge_samples.append(recycled_edge_samples[:needed])

            # Populate the new plate

            samples_to_place = list(self.samples)

            # Remove all edge-assigned samples from the interior pool (PRE-EMPTIVE REMOVAL)
            for sample in edge_placements:
                if sample in samples_to_place:
                    samples_to_place.remove(sample)

        random.shuffle(samples_to_place)

        temp_edge_placements = list(edge_placements)
        random.shuffle(temp_edge_placements)

        perimeter_indices_to_use = perimeter_indices
        if self.is_control_map_fixed:
            perimeter_indices_to_use = [
                idx for idx in perimeter_indices if idx not in self.fixed_control_map
            ]

        for r, c in perimeter_indices_to_use:
            if not temp_edge_placements:
                break

            sample = temp_edge_placements.pop()
            plate[r, c] = sample

            if sample in self.samples:
                self.used_edge_samples.add(sample)

        interior_indices_to_use = interior_indices
        if self.is_control_map_fixed:
            interior_indices_to_use = [
                idx for idx in interior_indices if idx not in self.fixed_control_map
            ]

        for r, c in interior_indices_to_use:
            if not samples_to_place:
                break
            plate[r, c] = samples_to_place.pop()

        # Fix the control map
        if not self.is_control_map_fixed:
            for r in range(self.plate_dims[0]):
                for c in range(self.plate_dims[1]):
                    sample = plate[r, c]
                    if sample in self.control_samples:
                        self.fixed_control_map[(r, c)] = sample
            self.is_control_map_fixed = True

        return plate


class PlateMapperNeighborAware(BasePlateMapper):
    """Generate plate maps with neighbor-awareness to minimize re-neighboring.

    Attributes
    ----------
    neighbor_pairs : set[tuple[str, str]]
        A set of tuples, each containing a pair of neighboring samples.
    """

    def __init__(
        self,
        sample_ids: list[str],
        control_sample_ids: list[str],
        plate_size: int = 384,
        predefined_control_map=None,
        nonstandard=False,
        nonstandard_dims=None,
        initial_position_map=None,
    ):
        super().__init__(
            sample_ids,
            control_sample_ids,
            plate_size,
            predefined_control_map,
            nonstandard,
            nonstandard_dims,
            initial_position_map,
        )
        self.neighbor_pairs: set[tuple[str, str]] = set()

    def _get_neighbors(self, r: int, c: int, plate: np.ndarray) -> list[str]:
        """Get the existing, non-empty neighbors of a given well.

        Parameters
        ----------
        r : int
            The row index of the well.
        c : int
            The column index of the well.
        plate : np.ndarray
            The plate map.

        Returns
        -------
        list[str]
            A list of the sample IDs of the neighbors.
        """
        neighbors = []
        rows, cols = self.plate_dims

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and plate[nr, nc] is not None:
                neighbors.append(plate[nr, nc])
        return neighbors

    def _update_neighbor_state(self, plate: np.ndarray):
        """Scan a completed plate and add all the neighbor pairs to the state.

        Parameters
        ----------
        plate : np.ndarray
            The plate map to scan.
        """
        rows, cols = self.plate_dims
        for r in range(rows):
            for c in range(cols):
                current_sample = plate[r, c]
                if current_sample is None:
                    continue
                # Right
                if c + 1 < cols and plate[r, c + 1] is not None:
                    neighbor_sample = plate[r, c + 1]
                    # Store as a sorted tuple to treat (A, B) and (B, A) as same pair
                    pair = tuple(sorted((current_sample, neighbor_sample)))
                    self.neighbor_pairs.add(pair)
                # Below
                if r + 1 < rows and plate[r + 1, c] is not None:
                    neighbor_sample = plate[r + 1, c]
                    # Store as a sorted tuple to treat (A, B) and (B, A) as same pair
                    pair = tuple(sorted((current_sample, neighbor_sample)))
                    self.neighbor_pairs.add(pair)

    def generate_plate(self) -> np.ndarray:
        """Generate a single plate using a constrained randomization approach.

        Returns
        -------
        np.ndarray
            A 2D numpy array representing the plate map.
        """
        plate = np.full(self.plate_dims, None, dtype=object)
        recycled_on_this_plate = []

        if self.is_control_map_fixed:
            # Place controls first
            for (r, c), sample in self.fixed_control_map.items():
                plate[r, c] = sample

            all_indices = {
                (r, c) for r in range(self.plate_dims[0]) for c in range(self.plate_dims[1])
            }
            fixed_indices = set(self.fixed_control_map.keys())

            variable_indices = list(all_indices - fixed_indices)
            random.shuffle(variable_indices)

            samples_to_place = list(self.samples)

        else:
            variable_indices = [
                (r, c) for r in range(self.plate_dims[0]) for c in range(self.plate_dims[1])
            ]
            random.shuffle(variable_indices)

            samples_to_place = list(self.all_samples)

        for r, c in variable_indices:
            is_edge = r == 0 or r == self.plate_dims[0] - 1 or c == 0 or c == self.plate_dims[1] - 1
            # Find best suitable sample for current well
            best_candidate = None

            candidates = list(samples_to_place)
            random.shuffle(candidates)

            for candidate_sample in candidates:
                # Check Neighbors
                neighbors = self._get_neighbors(r, c, plate)
                has_bad_neighbor = False
                for neighbor in neighbors:
                    pair = tuple(sorted((candidate_sample, neighbor)))
                    if pair in self.neighbor_pairs:
                        has_bad_neighbor = True
                        break
                # Try next candidate if sample has had neighbor
                if has_bad_neighbor:
                    continue

                is_variable_sample = candidate_sample in self.samples

                # Check if it has been used on an edge
                if (
                    is_edge
                    and is_variable_sample
                    and candidate_sample not in self.used_edge_samples
                ):
                    best_candidate = candidate_sample
                    break  # Use this one

                # If not an edge, or can't find unused edge sample, take first candidate with no reused neighbor
                if best_candidate is None:
                    best_candidate = candidate_sample

            # If fails on constraints, fall back to normal placement
            if best_candidate is None:
                click.echo(
                    f"Warning: Could not find 'perfect' sample for position ({r}, {c}). Placing first available sample."
                )
                if not samples_to_place:
                    click.echo("No more samples!")
                    continue
                best_candidate = samples_to_place[0]

            # After all that checking, we place the sample...
            plate[r, c] = best_candidate
            samples_to_place.remove(best_candidate)
            if is_edge and best_candidate in self.samples:
                if best_candidate in self.used_edge_samples:
                    recycled_on_this_plate.append(best_candidate)
                self.used_edge_samples.add(best_candidate)

        if recycled_on_this_plate:
            self.multi_edge_samples.append(recycled_on_this_plate)

        # Fix up the control map if not done so already.
        if not self.is_control_map_fixed:
            for r in range(self.plate_dims[0]):
                for c in range(self.plate_dims[1]):
                    sample = plate[r, c]
                    if sample in self.control_samples:
                        self.fixed_control_map[(r, c)] = sample

            self.is_control_map_fixed = True

        # We should now update all the neighbor pairs
        self._update_neighbor_state(plate)
        return plate
