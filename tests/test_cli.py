"""Pytest suite for the wellshuffled command line interface."""

import re

import pytest
from click.testing import CliRunner

from wellshuffled.cli import wellshuffled


@pytest.fixture
def oversized_sample_file(tmp_path):
    """Create a sample file holding more samples than a 96-well plate can hold."""
    p = tmp_path / "too_many_samples.txt"
    p.write_text("\n".join(f"sample-{i + 1}" for i in range(100)))
    return str(p)


def test_shuffle_and_trace_support_a_plate_with_more_than_26_rows(tmp_path):
    """Trace labels every row of a full 32-row plate A..Z then AA..AF."""
    sample_file = tmp_path / "samples.txt"
    # Filling every well makes the expected set of row labels seed-independent.
    sample_file.write_text("\n".join(f"sample-{i + 1}" for i in range(32 * 48)))
    plate_map = tmp_path / "plate_map.csv"
    trace_csv = tmp_path / "trace.csv"

    shuffled = CliRunner().invoke(
        wellshuffled,
        [
            "shuffle",
            str(sample_file),
            str(plate_map),
            "--nonstandard",
            "--nonstandard_dims",
            "32,48",
            "--seed",
            "1",
            "--simple",
        ],
    )
    assert shuffled.exit_code == 0, shuffled.output

    traced = CliRunner().invoke(
        wellshuffled, ["trace", str(plate_map), "--output-csv", str(trace_csv)]
    )
    assert traced.exit_code == 0, traced.output

    # Row labels must be letters only, never the punctuation chr() used to produce.
    row_labels = set()
    for line in trace_csv.read_text().splitlines()[1:]:
        match = re.search(r",([A-Za-z]+)\d+$", line)
        assert match, f"unparseable trace row: {line}"
        row_labels.add(match.group(1))

    expected = {chr(ord("A") + i) for i in range(26)} | {f"A{chr(ord('A') + i)}" for i in range(6)}
    assert row_labels == expected


@pytest.mark.parametrize(
    "rows",
    [
        ["sample-1,A1", "sample-2,A1"],
        ["sample-1,A1", "sample-2,a1"],
    ],
)
def test_shuffle_reports_clean_error_for_a_duplicate_well_in_the_sample_file(tmp_path, rows):
    """A sample file placing two samples in one well reports a usage error, not a traceback."""
    sample_file = tmp_path / "samples.csv"
    sample_file.write_text("\n".join(rows))

    result = CliRunner().invoke(
        wellshuffled, ["shuffle", str(sample_file), str(tmp_path / "out.csv")]
    )

    assert result.exit_code == 2, result.output
    assert not isinstance(result.exception, ValueError)
    assert "duplicated" in result.output


@pytest.mark.parametrize("well", ["ı1", "ſ1"])
def test_shuffle_rejects_non_ascii_well_in_sample_file(tmp_path, well):
    """A sample-file well that Unicode upper-casing would turn into a real well (ı1 -> I1) is rejected.

    A 20-row plate makes both I1 and S1 (from ſ1) real wells, so rejection indicates the
    non-ASCII input rather than an out-of-bounds row.
    """
    sample_file = tmp_path / "samples.csv"
    sample_file.write_text(f"sample-1,{well}\nsample-2,A2")
    out = tmp_path / "out.csv"

    result = CliRunner().invoke(
        wellshuffled,
        ["shuffle", str(sample_file), str(out), "--nonstandard", "--nonstandard_dims", "20,24"],
    )

    assert result.exit_code == 2, result.output
    assert not out.exists()


@pytest.mark.parametrize("well", ["ı1", "ſ1"])
def test_shuffle_rejects_non_ascii_well_in_fixed_map(tmp_path, well):
    """A --fixed-map well that Unicode upper-casing would turn into a real well is rejected."""
    sample_file = tmp_path / "samples.txt"
    sample_file.write_text("sample-1\ncontrol-1")
    out = tmp_path / "out.csv"

    result = CliRunner().invoke(
        wellshuffled,
        [
            "shuffle",
            str(sample_file),
            str(out),
            "--nonstandard",
            "--nonstandard_dims",
            "20,24",
            "--control-prefix",
            "control-",
            "--fixed-map",
            f"{well}:control-1",
        ],
    )

    assert result.exit_code == 2, result.output
    assert not out.exists()


@pytest.mark.parametrize("well", ["ı1", "ſ1"])
def test_shuffle_rejects_non_ascii_well_in_fixed_map_file(tmp_path, well):
    """A --fixed-map-file well that Unicode upper-casing would turn into a real well is rejected."""
    sample_file = tmp_path / "samples.txt"
    sample_file.write_text("sample-1\ncontrol-1")
    map_file = tmp_path / "controls.csv"
    map_file.write_text(f"WellPos,SampleID\n{well},control-1")
    out = tmp_path / "out.csv"

    result = CliRunner().invoke(
        wellshuffled,
        [
            "shuffle",
            str(sample_file),
            str(out),
            "--nonstandard",
            "--nonstandard_dims",
            "20,24",
            "--control-prefix",
            "control-",
            "--fixed-map-file",
            str(map_file),
        ],
    )

    assert result.exit_code == 2, result.output
    assert not out.exists()


@pytest.mark.parametrize("initial_well", ["1", "A01"])
def test_shuffle_reports_conflict_between_sample_file_and_fixed_map_across_spellings(
    tmp_path, initial_well
):
    """A sample and a fixed control in the same well (spelled differently) is a conflict, not a silent drop."""
    sample_file = tmp_path / "samples.csv"
    sample_file.write_text(f"sampleX,{initial_well}\ncontrol-1")
    out = tmp_path / "out.csv"

    result = CliRunner().invoke(
        wellshuffled,
        [
            "shuffle",
            str(sample_file),
            str(out),
            "--control-prefix",
            "control-",
            "--fixed-map",
            "A1:control-1",
            "--seed",
            "1",
        ],
    )

    assert result.exit_code == 2, result.output
    assert "different" in result.output
    assert not out.exists()


def test_shuffle_reports_clean_error_for_a_non_control_in_the_fixed_map(tmp_path):
    """A fixed map naming a sample that is not a control reports a usage error, not a traceback."""
    sample_file = tmp_path / "samples.txt"
    sample_file.write_text("\n".join([*[f"sample-{i}" for i in range(1, 20)], "control-1"]))

    result = CliRunner().invoke(
        wellshuffled,
        [
            "shuffle",
            str(sample_file),
            str(tmp_path / "out.csv"),
            "--control-prefix",
            "control-",
            "--fixed-map",
            "A2:sample-5",
        ],
    )

    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_shuffle_reports_clean_error_for_a_well_outside_the_plate(tmp_path):
    """A fixed map naming a well the plate does not have reports a usage error, not a traceback."""
    sample_file = tmp_path / "samples.txt"
    sample_file.write_text("\n".join([*[f"sample-{i}" for i in range(1, 20)], "control-1"]))

    result = CliRunner().invoke(
        wellshuffled,
        [
            "shuffle",
            str(sample_file),
            str(tmp_path / "out.csv"),
            "--nonstandard",
            "--nonstandard_dims",
            "32,48",
            "--control-prefix",
            "control-",
            "--fixed-map",
            "ZZ99:control-1",
        ],
    )

    assert result.exit_code == 2
    assert "Max well is AF48" in result.output
    assert "Traceback" not in result.output


def test_shuffle_reports_clean_error_when_samples_exceed_plate(oversized_sample_file, tmp_path):
    """Shuffle reports a usage error rather than a traceback when samples exceed plate capacity."""
    result = CliRunner().invoke(
        wellshuffled,
        ["shuffle", oversized_sample_file, str(tmp_path / "out.csv"), "--size", "96"],
    )

    assert result.exit_code == 2
    assert "exceeds" in result.output
    assert "Traceback" not in result.output
