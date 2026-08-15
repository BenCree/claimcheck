"""
Rule test code for unit testing of rules generated with Snakemake 9.25.1.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from subprocess import check_output

sys.path.insert(0, os.path.dirname(__file__))


def test_check_claims(conda_prefix):

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir) / "workdir"
        config_path = Path(".tests/unit/check_claims/config")
        data_path = Path(".tests/unit/check_claims/data")
        expected_path = Path(".tests/unit/check_claims/expected")

        # Copy config to the temporary workdir.
        shutil.copytree(config_path, workdir)

        # Copy data to the temporary workdir.
        shutil.copytree(data_path, workdir, dirs_exist_ok=True)

        # Run the test job.
        check_output(
            [
                sys.executable,
                "-m",
                "snakemake",
                "results/claims.json",
                "--snakefile",
                "Snakefile",
                "-f",
                "--notemp",
                "--show-failed-logs",
                "-j1",
                "--target-files-omit-workdir-adjustment",
                "--allowed-rules",
                "check_claims",
                "--directory",
                workdir,
            ]
            + conda_prefix
        )

        # Check the output byte by byte using cmp/zmp/bzcmp/xzcmp.
        # To modify this behavior, you can inherit from common.OutputChecker in here
        # and overwrite the method `compare_files(generated_file, expected_file),
        # also see common.py.
        import common
        common.OutputChecker(data_path, expected_path, workdir).check()
