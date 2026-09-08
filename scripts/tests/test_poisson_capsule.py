import importlib.util
import json
import os
from pathlib import Path
import tempfile
import sys
import subprocess
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'Docker_config/poisson' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


capsule = module('capsule')
archive = module('archive')


class CapsuleTests(unittest.TestCase):
    def test_launcher_uses_explicit_context_and_separate_hip_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docker = root / 'fake-docker'
            docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$CAPSULE_COMMAND_LOG"\nprintf "[]\\n"\n')
            docker.chmod(0o755)
            log = root / 'commands'
            env = dict(os.environ, CAPSULE_DOCKER=str(docker), CAPSULE_COMMAND_LOG=str(log),
                       CAPSULE_DOCKER_CONTEXT='isolated-rootless', CAPSULE_HIP_VISIBLE_DEVICES='0',
                       CAPSULE_UCX_TLS='self,sm,tcp,rocm_copy')
            subprocess.run(['bash', str(ROOT / 'Docker_config/poisson/run.sh'), 'hip', 'capsule:test',
                            str(root / 'results'), 'verify', '--ranks', '1'],
                           env=env, check=True, stdout=subprocess.PIPE)
            command = log.read_text().splitlines()
            self.assertIn('isolated-rootless', command)
            self.assertIn('/dev/dri/renderD128', command)
            self.assertIn('/dev/dri/renderD129', command)
            self.assertIn('ROCR_VISIBLE_DEVICES=0', command)
            self.assertIn('CAPSULE_UCX_TLS=self,sm,tcp,rocm_copy', command)
            self.assertNotIn('--privileged', command)
            self.assertNotIn('--ipc=host', command)
            self.assertNotIn('use', command)

    def test_mpi_support_is_required_at_build_time(self):
        script = (ROOT / 'Docker_config/poisson/build_mpi.sh').read_text()
        self.assertIn('--with-cuda-libdir=/usr/local/cuda/lib64/stubs', script)
        self.assertIn('OPAL_CUDA_SUPPORT +1', script)
        self.assertIn('OPAL_ROCM_SUPPORT +1', script)
        probe = (ROOT / 'Docker_config/poisson/probe.cpp').read_text()
        self.assertIn('MPI_Alltoallv(', probe)
        self.assertIn('MPI_COMM_SELF', probe)

    def test_runner_preserves_failure_and_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'metadata').mkdir()
            with mock.patch.object(capsule, 'PREFIX', root):
                runner = capsule.Runner(root / 'output', 'cuda', 0.1)
                def validate(rc, text):
                    if rc != 0:
                        raise ValueError('nonzero exit')
                    return dict(ok=True)
                self.assertFalse(runner.execute('exit', [sys.executable, '-c',
                                                        'print("debug detail"); raise SystemExit(7)'], {}, validate))
                self.assertFalse(runner.execute('timeout', [sys.executable, '-c',
                                                           'import time; time.sleep(10)'], {}, validate))
                self.assertEqual(runner.summary(), 1)
                self.assertEqual(runner.records[0]['returncode'], 7)
                self.assertEqual(runner.records[1]['returncode'], 124)
                self.assertIn('debug detail', (root / 'output/exit.log').read_text())
                with self.assertRaisesRegex(ValueError, 'already'):
                    capsule.Runner(root / 'output', 'cuda', 1)

    def test_results(self):
        text = 'INFO poisson_periodic_4d_autotuned: size=16x20x24x32 mpi=2 rel_l2=2e-16 source=cache'
        self.assertEqual(capsule.result_fields(text, 4)['source'], 'cache')
        for bad in ('nan', 'inf', '-1', '1e-4'):
            with self.assertRaises(ValueError):
                capsule.result_fields(text.replace('2e-16', bad), 4)
        for bad in ('', text + '\n' + text):
            with self.assertRaises(ValueError):
                capsule.result_fields(bad, 4)

    def test_transport_isolation(self):
        with mock.patch.dict('os.environ', {'FFTM_USE_FFT_EXEC_NO_SYNC': '1', 'UCX_TLS': 'bad'}):
            aware = capsule.mpi_environment('device-aware')
            staged = capsule.mpi_environment('host-staged')
        self.assertNotIn('FFTM_USE_FFT_EXEC_NO_SYNC', aware)
        self.assertNotIn('UCX_TLS', aware)
        self.assertEqual(aware['OMPI_MCA_pml'], 'ucx')
        self.assertEqual(staged['OMPI_MCA_pml'], 'ob1')
        self.assertNotIn('OMPI_MCA_pml_ucx_tls', staged)

    def test_explicit_gpu_transport_override(self):
        with mock.patch.dict('os.environ', {'CAPSULE_UCX_TLS': 'self,sm,tcp,rocm_copy'}):
            aware = capsule.mpi_environment('device-aware')
            staged = capsule.mpi_environment('host-staged')
        self.assertEqual(aware['UCX_TLS'], 'self,sm,tcp,rocm_copy')
        self.assertNotIn('UCX_TLS', staged)
        dockerfile = (ROOT / 'Docker_config/poisson/Dockerfile.hip').read_text()
        self.assertIn('ENV CAPSULE_UCX_TLS=self,sm,tcp,rocm_copy', dockerfile)

    def test_archive_checks_order_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'image.tar.gz'
            source.write_bytes(b'0123456789abcdefghijklmnopqrstuvwxyz')
            manifest = archive.split_archive(source, root, part_bytes=13)
            path = root / 'manifest.json'
            path.write_text(json.dumps(manifest))
            self.assertEqual(len(archive.verify(path)[1]), 3)
            manifest['parts'].reverse()
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'combined'):
                archive.verify(path)
            manifest['parts'].reverse()
            path.write_text(json.dumps(manifest))
            first = root / manifest['parts'][0]['name']
            first.write_bytes(b'x' * first.stat().st_size)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                archive.verify(path)
            first.unlink()
            with self.assertRaises(FileNotFoundError):
                archive.verify(path)

    def test_archive_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text(json.dumps(dict(schema=1, parts=[dict(name='../outside')], archive_sha256='')))
            with self.assertRaisesRegex(ValueError, 'name'):
                archive.verify(path)


if __name__ == '__main__':
    unittest.main()
