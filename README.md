# WellShuffled 🎲

-----

## Installation

This project is managed with `uv`, a fast Python package installer and resolver.

1.  **Clone the repository:**

    ```
    git clone https://gitlab.com/mcnaughtonadm/wellshuffled.git
    cd wellshuffled
    ```

2.  **Sync project using `uv`:**

    ```
    uv sync
    ```
	
3. **Installing without `uv sync`**

	If `uv` has errors during sync you can install manually with `pip`:

	```
	uv run python -m pip install -e .
	```
	
	or
	
	```
	python -m pip install -e .
	```
	

-----

## Usage

The main command is `wellshuffled`. It takes a required input file of sample IDs and an output path.

### Basic Command

```
uv run wellshuffled <sample_file> <output_path> [OPTIONS]
```

  * `SAMPLE_FILE`: A text file with one sample ID per line.
  * `OUTPUT_PATH`: The path for the output CSV file or directory.

### Examples

**1. Generate a single 96-well plate with neighbor-awareness:**

```
uv run wellshuffled samples.csv single_plate.csv --size 96
```

**2. Generate 5 plates and save to a single combined file:**

```
uv run wellshuffled samples.csv combined_layouts.csv --plates 5
```

**3. Generate 3 plates and save them to separate files in a directory:**
The output path `my_layouts` will be created if it doesn't exist.

```
uv run wellshuffled samples.csv my_layouts --plates 3 --separate-files
```

**4. Generate a reproducible layout using a seed:**
Running this command multiple times with `--seed 123` will always produce the exact same output file.

```
uv run wellshuffled samples.csv reproducible_run.csv --plates 4 --seed 123
```

**5. Use the simple randomization logic (disabling neighbor-awareness):**

```
uv run wellshuffled samples.csv simple_run.csv --simple
```

**6. View all available options and help:**

```
uv run wellshuffled --help
```
