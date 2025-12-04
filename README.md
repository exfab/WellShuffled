# WellShuffled 🎲

---

## Installation

There are two ways to install `wellshuffled`:

**Option 1: Install from GitLab using `pip`**

You can install the latest version of `wellshuffled` directly from GitLab by
running the following command:

```
pip install git+https://github.com/exfab/wellshuffled.git
```

**Option 2: Install from a local clone**

1.  **Clone the repository:**

    ```
    git clone https://github.com/exfab/wellshuffled.git
    cd wellshuffled
    ```

2.  **Install using `pip`:**

    ```
    pip install .
    ```

---

## Usage

The main command is `wellshuffled`. It has two subcommands: `shuffle` and
`trace`.

After installation, you can run the tool from your terminal like this:

```
wellshuffled --help
```

### `shuffle` Command

The `shuffle` command generates randomized plate maps from a list of sample IDs.

```
wellshuffled shuffle <sample_file> <output_path> [OPTIONS]
```

**Options:**

- `--plates, -n`: Number of plates to generate. Default is 1.
- `--size`: Well plate size (96 or 384). Default is 96.
- `--simple`: Use simple randomization (disables neighbor-awareness).
- `--separate-files`: Save each plate map to a separate CSV file.
- `--seed`: Set the random seed for reproducible results.
- `--control-prefix`: Prefix used to identify control samples.
- `--fixed-map`: Manually specify fixed control locations (e.g.,
  "A1:control-85,H12:control-96").
- `--fixed-map-file`: Manually specify fixed control locations from a CSV file.
- `--nonstandard`: Allow for the use of non-standard plate dimensions.
- `--nonstandard_dims`: The dimensions (x, y) for the nonstandard plate.

### `trace` Command

The `trace` command traces the locations of samples across multiple plates.

```
wellshuffled trace <input_path> [OPTIONS]
```

**Options:**

- `--output-csv`: Optional path to save the full trajectory map as a CSV file.
- `--numeric`: Return the plate positions as 1-based column major numeric
  values.

### Examples

**1. Generate a single 96-well plate with neighbor-awareness:**

```
wellshuffled shuffle example_files/samples.csv single_plate.csv --size 96
```

**2. Generate 5 plates and save to a single combined file:**

```
wellshuffled shuffle example_files/samples.csv combined_layouts.csv --plates 5
```

**3. Generate 3 plates and save them to separate files in a directory:** The
output path `my_layouts` will be created if it doesn't exist.

```
wellshuffled shuffle example_files/samples.csv my_layouts --plates 3 --separate-files
```

**4. Generate a reproducible layout using a seed:** Running this command
multiple times with `--seed 123` will always produce the exact same output file.

```
wellshuffled shuffle example_files/samples.csv reproducible_run.csv --plates 4 --seed 123
```

**5. Use the simple randomization logic (disabling neighbor-awareness):**

```
wellshuffled shuffle example_files/samples.csv simple_run.csv --simple
```

**6. Generate a plate with a predefined layout for the first plate:** The
`samples_with_positions.csv` file contains a second column with the initial well
positions.

```
wellshuffled shuffle example_files/samples_with_positions.csv predefined_layout.csv
```

**7. Provide Nonstandard plate dimensions:** We use the `--nonstandard` flag to
ensure you are explicitly meaning to pass in nonstandard plate dimensions along
with the `--nonstandard_dims=[RowsxColumns] or [Rows,Columns]`

```
wellshuffled shuffle example_files/60_samples_w_controls_initial_pos.csv nonstandard_layout.csv --nonstandard --nonstandard_dims=6,10
```

**8. View all available options and help:**

```
wellshuffled --help
```

## Development Installation

If you want to contribute to the development of `wellshuffled`, you can set up a
development environment by following these steps:

1.  **Clone the repository:**

    ```
    git clone https://github.com/exfab/wellshuffled.git
    cd wellshuffled
    ```

2.  **Sync the project using `uv`:**

    This project is managed with `uv`, a fast Python package installer and
    resolver.

    ```
    uv sync
    ```

3.  **Installing without `uv sync` or using `venv`**

    If you encounter errors with `uv sync`, you can install the project manually
    in editable mode:

    Using `uv`:

    ```
    uv run python -m pip install -e .
    ```

    Using `venv`:

    ```
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -e .
    ```
