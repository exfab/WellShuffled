"""Script for generating randomized plates for 96 or 384-well configurations."""

import random
from abc import ABC, abstractmethod

import numpy as np

from wellshuffled.utilities import (
    well_to_index,
)


class BasePlateMapper(ABC):
    """Base class for shared PlateMapper functionality."""

    def __init__(
        self,
        sample_ids: list[str],
        control_sample_ids: list[str],
        plate_size: int = 96,
        predefined_control_map: dict[str, str] | None = None,
    ):
        if plate_size not in [96, 384]:
            raise ValueError("Plate size unknown, must be 96 or 384.")

        self.samples = list(sample_ids)
        self.control_samples = list(control_sample_ids)
        self.all_samples = self.samples + self.control_samples

        self.plate_size = plate_size
        self.plate_dims = (16, 24) if plate_size == 384 else (8, 12)

        self.fixed_control_map: dict[tuple[int, int], str] = {}
        self.is_control_map_fixed = False

        rows, cols = self.plate_dims
        all_indices = rows * cols

        if len(self.all_samples) > all_indices:
            raise ValueError(
                f"Total Samples ({len(self.all_samples)}) exceeds wells ({all_indices})."
            )

        # Track state of edge samples, and sample neighbors
        self.used_edge_samples: set[str] = set()
        self.multi_edge_samples: list[list[str]] = []

        if predefined_control_map:
            # Manual control map provided
            if len(predefined_control_map) != len(self.control_samples):
                raise ValueError(
                    "The number of samples in the fixed map must match the number of control samples."
                )

            # Convert the WELL:SAMPLE_ID map to (R,C):SAMPLE_ID map
            for well, sample_id in predefined_control_map.items():
                r, c = well_to_index(well, self.plate_dims)
                self.fixed_control_map[(r, c)] = sample_id

            self.is_control_map_fixed = True

        elif not self.control_samples:
            # If no controls, no map needed.
            self.is_control_map_fixed = True

    @abstractmethod  # Enforces that this method must be implemented by subclasses
    def generate_plate(self) -> np.ndarray:
        """Abstract method to generate a single plate map. Must be implemented by subclasses."""
        pass

    def _get_perimeter_indices(self) -> tuple[list[tuple], list[tuple]]:
        """Calculate the (row, col) indices for perimeter and interior well positions."""
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
        """Generate a specified number of unique plate layouts."""
        return [self.generate_plate() for _ in range(num_plates)]  # TODO: Fix this line


class PlateMapperSimple(BasePlateMapper):
    """Generates and manages randomized plate maps for plate dimensions w/o neighbor-awareness."""

    def generate_plate(self) -> np.ndarray:
        """Generate a single randomized plate map."""
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
    """Generate Plate maps with neighbor-awareness to minimize re-neighboring."""

    def __init__(
        self,
        sample_ids: list[str],
        control_sample_ids: list[str],
        plate_size: int = 384,
        predefined_control_map=None,
    ):
        super().__init__(sample_ids, control_sample_ids, plate_size, predefined_control_map)
        self.neighbor_pairs: set[tuple[str, str]] = set()

    def _get_neighbors(self, r: int, c: int, plate: np.ndarray) -> list[str]:
        """Get the existing, non-empty neighbors of a given well."""
        neighbors = []
        rows, cols = self.plate_dims

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and plate[nr, nc] is not None:
                neighbors.append(plate[nr, nc])
        return neighbors

    def _update_neighbor_state(self, plate: np.ndarray):
        """Scan a completed plate and add all the neighbor pairs to the state."""
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
        """Generate a single plate using a constrained randomization approach."""
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
                print(
                    f"Warning: Could not find 'perfect' sample for position ({r}, {c}). Placing first available sample."
                )
                if not samples_to_place:
                    print("No more samples!")
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

    def generate_multiple_plates(self, num_plates: int) -> list[np.ndarray]:
        """Generate a specified number of unique plate layouts."""
        return [self.generate_plate() for _ in range(num_plates)]
