import pytest
import subprocess
from pathlib import Path
import numpy as np
import json
import numpy.testing as npt
import os

ROOT_DIR = Path(__file__).parent.parent
OUTPUT_PATH = ROOT_DIR / "gpe_test_output"
SNAPSHOT_PATH = ROOT_DIR / "tests/snapshots/gpe_snapshot"

RTOL = 1e-5
ATOL = 1e-8


@pytest.fixture(autouse=True, scope="session")
def apply_test_override():
    """
    Ensures that configOverrides.json begins with the test override block
    before any tests run.
    """
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    data = {
        "iterations": 8192,
        "gridWidth": 512,
        "gridHeight": 512,
        "threadsPerBlockX": 32,
        "threadsPerBlockY": 32,
        "downloadFrequency": 1024,
        "simulationMode": "grossPitaevskii",
        "grossPitaevskii": {
            "L": 1.0,
            "x0": 0.15,
            "y0": 0.15,
            "kx": 0,
            "ky": 0,
            "sigma": 0.1,
            "amp": 1.0,
            "trapStr": 5e6,
            "V_bias": 500,
            "r_0": 0.05,
            "sigma2": 0.025,
            "absorbStrength": 10e5,
            "absorbWidth": 0.025,
            "dt": 6e-7,
            "g": 10e1,
        },
    }

    with open("configOverrides.json", "w") as f:
        json.dump(data, f, indent=4)

    subprocess.run(["./bin/main", "--output", str(OUTPUT_PATH)], check=True)


def test_wavefunction_evolution_fidelity():
    """
    Validates the simulation output against a reference snapshot to ensure numerical fidelity.

    This test performs the following verifications:
    1. Metadata Consistency: Checks that simulation dimensions, iteration
    counts, and download frequencies match the snapshot header.
    2. Data Integrity: Compares the complex wavefunction data slice-by-slice
    against the snapshot using standard tolerance.
    3. Physical Validity: Asserts that the wavefunction remains normalized (L2
    norm ≈ 1.0) at every time step to ensure probability conservation.
    """
    with open(SNAPSHOT_PATH, "rb") as snapshot_file:
        with open(OUTPUT_PATH, "rb") as output_file:
            header_line = snapshot_file.readline()
            snapshot_header_data = json.loads(header_line)

            snapshot_width = int(snapshot_header_data["width"])
            snapshot_height = int(snapshot_header_data["height"])
            snapshot_iterations = int(snapshot_header_data["iterations"])
            snapshot_downloadFrequency = int(snapshot_header_data["downloadFrequency"])
            snapshot_dx = float(snapshot_header_data["parameterData"]["dx"])
            snapshot_dy = float(snapshot_header_data["parameterData"]["dy"])

            header_line = output_file.readline()
            output_header_data = json.loads(header_line)

            output_width = int(output_header_data["width"])
            output_height = int(output_header_data["height"])
            output_iterations = int(output_header_data["iterations"])
            output_downloadFrequency = int(output_header_data["downloadFrequency"])
            output_dx = float(output_header_data["parameterData"]["dx"])
            output_dy = float(output_header_data["parameterData"]["dy"])

            assert output_width == snapshot_width, "widths must be equal"
            assert output_height == snapshot_height, "heights must be equal"
            assert output_iterations == snapshot_iterations, "heights must be equal"
            assert (
                output_downloadFrequency == snapshot_downloadFrequency
            ), "downloadFrequency must be equal"
            assert output_dx == snapshot_dx
            assert output_dy == snapshot_dy

            width = output_width
            height = output_height
            slice_size = width * height
            dx = output_dx
            dy = output_dy
            current_iter = 0
            max_iter = output_iterations // output_downloadFrequency

            while current_iter < max_iter:
                snapshot_flat_slice = np.fromfile(
                    snapshot_file, dtype=np.complex64, count=slice_size
                )
                output_flat_slice = np.fromfile(
                    output_file, dtype=np.complex64, count=slice_size
                )

                snapshot_array_2d = snapshot_flat_slice.reshape((height, width))
                output_array_2d = output_flat_slice.reshape((height, width))

                npt.assert_allclose(
                    snapshot_array_2d,
                    output_array_2d,
                    rtol=RTOL,
                    atol=ATOL,
                    err_msg=f"Arrays differ at iteration {current_iter}",
                )

                probability_sum = np.sum(np.abs(snapshot_array_2d) ** 2) * (dx * dy)
                npt.assert_allclose(
                    probability_sum,
                    1.0,
                    rtol=RTOL,
                    err_msg="Total probability is not 1",
                )

                current_iter += 1
