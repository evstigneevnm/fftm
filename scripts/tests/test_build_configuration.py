#!/usr/bin/env python3
"""Exercise Make configuration/depfiles with isolated, GPU-free compiler fixtures."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


REPO = Path(__file__).resolve().parents[2]


class BuildConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fftm_make_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("build_configs",):
            shutil.copytree(REPO / name, self.root / name,
                            ignore=shutil.ignore_patterns("config_local*.inc"))
        for name in ("examples/Makefile", "examples/poisson/Makefile", "source/tests/Makefile"):
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / name, destination)
        self.inc = self.root / "source/detail/sample.inc"
        self.inc.parent.mkdir(parents=True)
        self.inc.write_text("// dependency fixture\n")
        for name in ("source/tests/test_fftm_options.cpp", "source/fftm_options.hpp",
                     "source/tests/test_fftm_3D_compare.cu",
                     "examples/poisson/source/poisson_periodic_3d_autotuned.cpp",
                     "examples/poisson/source/poisson_periodic_4d_autotuned.cpp"):
            file = self.root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("// source fixture\n")
        self.compiler = self.root / "compiler.py"
        self.compiler.write_text(
            "import json, os, pathlib, sys\n"
            "args = sys.argv[1:]\n"
            "with open(os.environ['COMPILER_LOG'], 'a') as f:\n"
            "    f.write(json.dumps(args) + '\\n')\n"
            "out = pathlib.Path(args[args.index('-o') + 1])\n"
            "out.write_text('fixture binary\\n')\n"
            "if '-MF' in args:\n"
            "    dep = pathlib.Path(args[args.index('-MF') + 1])\n"
            "    target = args[args.index('-MT') + 1]\n"
            "    dep.write_text(target + ': ' + os.environ['HEADER_FIXTURE'] + '\\n')\n"
        )
        self.log = self.root / "compiler.log"
        self.env = {k: v for k, v in os.environ.items()
                    if k in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")}
        self.env.update(COMPILER_LOG=str(self.log), HEADER_FIXTURE=str(self.inc))
        self.profile = self.root / "build_configs/test.inc"
        compiler = f"{sys.executable} {self.compiler}"
        self.profile.write_text(
            "FFTM_BUILD_BACKEND = cuda\n"
            "cuda_dir = /fixture/cuda\nmpi_dir = /fixture/mpi\n"
            "CUDA_ARCH_LIST = 70\nTARGET_NVCC = -O2\n"
            "BUILD_FOLDER = $(PROJECT_ROOT)/out\n"
            f"NVCC = {compiler}\nHIPCC = {compiler}\nMPICXX = {compiler}\nCXX = {compiler}\n"
        )

    def make(self, directory, *args, success=True):
        result = subprocess.run(
            ["make", "--no-print-directory", "-s", "-C", str(self.root / directory), *args],
            env=self.env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def count(self):
        return len(self.log.read_text().splitlines()) if self.log.exists() else 0

    def test_missing_and_empty_config(self):
        for value in ("", "missing.inc"):
            self.make("examples", f"CONFIG_FILE={value}", "print-config", success=False)
        empty = self.root / "empty.inc"
        empty.touch()
        self.make("source/tests", f"CONFIG_FILE={empty}", "print-config", success=False)

    def test_root_relative_absolute_and_command_override(self):
        for directory in ("examples", "examples/poisson", "source/tests"):
            output = self.make(directory, "CONFIG_FILE=build_configs/test.inc", "print-config",
                               "CUDA_ARCH_LIST=80", "TARGET_NVCC=-O1")
            self.assertIn("code=sm_80", output)
            self.assertIn("TARGET_NVCC=-O1", output)
            self.assertIn(f"CONFIG_FILE={self.profile}", output)
            absolute = self.make(directory, f"CONFIG_FILE={self.profile}", "print-config")
            self.assertIn("code=sm_70", absolute)

    def test_automatic_local_profile(self):
        shutil.copyfile(self.profile, self.root / "build_configs/config_local.inc")
        output = self.make("examples", "print-config")
        self.assertIn("cuda_dir=/fixture/cuda", output)

    def test_invalid_profile_values(self):
        for assignment in ("FFTM_BUILD_BACKEND=", "FFTM_BUILD_BACKEND=cuda hip",
                           "FFTM_BUILD_BACKEND=cpu", "FFTM_DEVICE_AWARE_MPI=2",
                           "FFTM_DEVICE_AWARE_MPI=0 1"):
            self.make("examples", "print-config", assignment, success=False)
        (self.root / "build_configs/config_local.inc").touch()
        self.make("examples", "print-config", success=False)

    def test_build_folder_precedence(self):
        with self.profile.open("a") as output:
            output.write("BUILD_DIR = $(PROJECT_ROOT)/profile-output\n")
        args = ("CONFIG_FILE=build_configs/test.inc", "print-config")
        output = self.make("examples", *args, f"BUILD_FOLDER={self.root}/cli-folder")
        self.assertIn(f"BUILD_DIR={self.root}/cli-folder", output)
        output = self.make("examples", *args, f"BUILD_FOLDER={self.root}/cli-folder",
                           f"BUILD_DIR={self.root}/cli-dir")
        self.assertIn(f"BUILD_DIR={self.root}/cli-dir", output)

    def test_mpi_overrides_and_depfile_in_test_macro(self):
        self.make("source/tests", "CONFIG_FILE=build_configs/test.inc",
                  "test_fftm_3D_compare.bin", "MPI_CPPFLAGS=-I/mpi/include1 -I/mpi/include2",
                  "MPI_LIBS=-lcustom_mpi -lmpi_extra", "MPI_LDFLAGS=-L/mpi/lib64")
        log = self.log.read_text()
        self.assertIn("-I/mpi/include2", log)
        self.assertIn("-lcustom_mpi", log)
        self.assertIn("-L/mpi/lib64", log)
        self.assertTrue((self.root / "out/test_fftm_3D_compare.bin.d").exists())

    def test_effective_capability_flags_trigger_rebuild(self):
        args = ("CONFIG_FILE=build_configs/test.inc", "test_fftm_3D_compare.bin")
        self.make("source/tests", *args)
        count = self.count()
        self.make("source/tests", *args, "SCFD_FLAGS_CA=-DTEST_ALTERNATE_CAPABILITY=1")
        self.assertGreater(self.count(), count)
        self.assertIn("-DTEST_ALTERNATE_CAPABILITY=1", self.log.read_text())

    def test_profile_beats_environment_command_beats_profile(self):
        self.env["cuda_dir"] = "/environment/cuda"
        output = self.make("examples", "CONFIG_FILE=build_configs/test.inc", "print-config")
        self.assertIn("cuda_dir=/fixture/cuda", output)
        output = self.make("examples", "CONFIG_FILE=build_configs/test.inc", "print-config",
                           "cuda_dir=/command/cuda")
        self.assertIn("cuda_dir=/command/cuda", output)

    def test_legacy_overrides_and_cpu_only(self):
        output = self.make("source/tests", "check-config-cxx", "CXX=true", "NVCC=/missing/nvcc")
        self.assertNotIn("not found", output)
        output = self.make("examples/poisson", "-n", "cuda", "CUDA_ARCH=-arch=sm_80",
                           f"BUILD_FOLDER={self.root}/legacy")
        self.assertIn("-arch=sm_80", output)
        self.assertIn(str(self.root / "legacy"), output)

    def test_recursive_configuration(self):
        self.make("examples", "CONFIG_FILE=build_configs/test.inc",
                  "poisson_periodic_3d_autotuned.bin", "CUDA_HOST_CXX=/fixture/g++")
        log = self.log.read_text()
        self.assertIn("-I/fixture/mpi/include", log)
        self.assertIn("/fixture/g++", log)
        count = self.count()
        self.make("examples", "CONFIG_FILE=build_configs/test.inc",
                  "poisson_periodic_3d_autotuned.bin", "CUDA_HOST_CXX=/fixture/g++")
        self.assertEqual(self.count(), count, "unchanged recursive build recompiled")

    def test_incremental_header_flags_and_profile_switch(self):
        args = ("CONFIG_FILE=build_configs/test.inc", "test_fftm_options.bin")
        self.make("source/tests", *args)
        self.assertEqual(self.count(), 1)
        self.make("source/tests", *args)
        self.assertEqual(self.count(), 1)
        time.sleep(0.02)
        self.inc.touch()
        self.make("source/tests", *args)
        self.assertEqual(self.count(), 2, "included .inc change was not rebuilt")
        self.make("source/tests", *args, "TARGET_GCC=-O0")
        self.assertEqual(self.count(), 3)
        self.make("source/tests", *args)
        self.assertEqual(self.count(), 4, "switching back to prior flags reused stale output")
        self.profile.write_text(self.profile.read_text().replace("CUDA_ARCH_LIST = 70", "CUDA_ARCH_LIST = 80"))
        self.make("source/tests", *args)
        self.assertEqual(self.count(), 5)

    def test_hip_default_and_transport_variants(self):
        self.make("examples", "CONFIG_FILE=build_configs/test.inc", "FFTM_BUILD_BACKEND=hip",
                  "FFTM_DEVICE_AWARE_MPI=0", "HIP_ARCH_LIST=gfx1102 gfx90a")
        log = self.log.read_text()
        self.assertIn("--offload-arch=gfx1102", log)
        self.assertIn("--offload-arch=gfx90a", log)
        self.assertNotIn("SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI", log)
        self.assertIn("_hip_nca.bin", log)
        self.make("examples/poisson", "CONFIG_FILE=build_configs/test.inc", "hip")
        self.assertIn("SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI", self.log.read_text())

    def test_cuda_only_target_rejected_in_hip_profile(self):
        self.make("examples", "FFTM_BUILD_BACKEND=hip", "taylor_green_spacetime_4d.bin", success=False)

    def test_single_quotes_in_flags(self):
        output = self.make("examples", "print-config", "CPPFLAGS=-DNAME='example'")
        self.assertIn("CPPFLAGS=-DNAME='example'", output)
        self.make("source/tests", "CONFIG_FILE=build_configs/test.inc",
                  "test_fftm_options.bin", "CPPFLAGS=-DNAME='example'")

    def test_parallel_build(self):
        self.make("examples", "CONFIG_FILE=build_configs/test.inc", "-j2",
                  "poisson_periodic_3d_autotuned.bin", "poisson_periodic_4d_autotuned.bin")
        self.assertTrue((self.root / "out/poisson_periodic_3d_autotuned.bin").exists())
        self.assertTrue((self.root / "out/poisson_periodic_4d_autotuned.bin").exists())


if __name__ == "__main__":
    unittest.main()
