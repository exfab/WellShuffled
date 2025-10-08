"""Script for generating randomized plates for 96 or 384-well configurations."""

import random

import numpy as np


class PlateMapperSimple:
    """Generates and manages randomized plate maps for plate dimensions w/o neighbor-awareness."""

    def __init__(self, sample_ids: list[str], plate_size: int = 384):
        if plate_size not in [96, 384]:
            raise ValueError("Plate size unknown, must be 96 or 384.")

        self.all_samples = list(sample_ids)
        self.plate_size = plate_size
        self.plate_dims = (16, 24) if plate_size == 384 else (8, 12)

        # Track state of edge samples, and sample neighbors
        self.used_edge_samples: set[str] = set()
        self.multi_edge_samples: list[list[str]] = []

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

    def generate_plate(self) -> np.ndarray:
        """Generate a single randomized plate map."""
        plate = np.full(self.plate_dims, None, dtype=object)
        perimeter_indices, interior_indices = self._get_perimeter_indices()
        num_perimeter_wells = len(perimeter_indices)

        # Find samples that have not been used on the edge yet.
        available_edge_samples = [s for s in self.all_samples if s not in self.used_edge_samples]
        random.shuffle(available_edge_samples)

        # Find samples that have already been used on an edge.
        recycled_edge_samples = list(self.used_edge_samples)
        random.shuffle(recycled_edge_samples)

        # Select samples for the edge
        edge_placements = available_edge_samples[:num_perimeter_wells]

        # If not enough fresh samples, supplement with already-used ones but keep track
        if len(edge_placements) < num_perimeter_wells:
            needed = num_perimeter_wells - len(edge_placements)
            edge_placements.extend(recycled_edge_samples[:needed])
            self.multi_edge_samples.append(recycled_edge_samples[:needed])

        # Populate the new plate

        samples_to_place = list(self.all_samples)
        random.shuffle(samples_to_place)

        temp_edge_placements = list(edge_placements)
        random.shuffle(temp_edge_placements)

        # Handle perimeter samples
        for r, c in perimeter_indices:
            sample = temp_edge_placements.pop()
            plate[r, c] = sample
            self.used_edge_samples.add(sample)  # Update the state of the used samples
            # Ensure we don't duplicate samples on the same plate
            if sample in samples_to_place:
                samples_to_place.remove(sample)

        # Handle interior samples
        for r, c in interior_indices:
            if not samples_to_place:
                break  # No more samples
            plate[r, c] = samples_to_place.pop()

        return plate

    def generate_multiple_plates(self, num_plates: int) -> list[np.ndarray]:
        """Generate a specified number of unique plate layouts."""
        return [self.generate_plate() for _ in range(num_plates)]

    @staticmethod
    def save_plate_to_csv(plate_data: np.ndarray, filename: str):
        """Save a given plate map (NumPy array) to a CSV file."""
        np.savetxt(filename, plate_data, delimiter=",", fmt="%s")
        print(f"Plate map successfully saved to {filename}")


class PlateMapperNeighborAware:
    """Generate Plate maps with neighbor-awareness to minimize re-neighboring."""

    def __init__(self, sample_ids: list[str], plate_size: int = 384):
        if plate_size not in [96, 384]:
            raise ValueError("Plate size unknown, must be 96 or 384.")

        self.all_samples = list(sample_ids)
        self.plate_size = plate_size
        self.plate_dims = (16, 24) if plate_size == 384 else (8, 12)

        # Track state of edge samples, and sample neighbors
        self.used_edge_samples: set[str] = set()
        self.multi_edge_samples: list[list[str]] = []
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

        samples_to_place = list(self.all_samples)
        random.shuffle(samples_to_place)

        all_indices = [(r, c) for r in range(self.plate_dims[0]) for c in range(self.plate_dims[1])]
        random.shuffle(all_indices)

        recycled_on_this_plate = []

        for r, c in all_indices:
            is_edge = r == 0 or r == self.plate_dims[0] - 1 or c == 0 or c == self.plate_dims[1] - 1
            # Find best suitable sample for current well
            best_candidate = None
            for candidate_sample in samples_to_place:
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

                # Check if it has been used on an edge
                if is_edge and candidate_sample not in self.used_edge_samples:
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
            if is_edge:
                if best_candidate in self.used_edge_samples:
                    recycled_on_this_plate.append(best_candidate)
                self.used_edge_samples.add(best_candidate)

        if recycled_on_this_plate:
            self.multi_edge_samples.append(recycled_on_this_plate)

        # We should now update all the neighbor pairs
        self._update_neighbor_state(plate)
        return plate

    def generate_multiple_plates(self, num_plates: int) -> list[np.ndarray]:
        """Generate a specified number of unique plate layouts."""
        return [self.generate_plate() for _ in range(num_plates)]

    @staticmethod
    def save_plate_to_csv(plate_data: np.ndarray, filename: str):
        """Save a given plate map (NumPy array) to a CSV file."""
        np.savetxt(filename, plate_data, delimiter=",", fmt="%s")
        print(f"Plate map successfully saved to {filename}")


def load_sample_ids(filename: str) -> list[str]:
    """Load in a list of file names from csv file."""
    data = []
    with open(filename, "r", newline="") as csv_file:
        for row in csv_file:
            data.append(row.strip())
    return data


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
