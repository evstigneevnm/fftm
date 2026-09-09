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
host_devices = module('host_devices')


class CapsuleTests(unittest.TestCase):
    def test_general_rank_selection(self):
        self.assertEqual(list(capsule.requested_ranks(None, '4')), [1, 2, 3, 4])
        self.assertEqual(capsule.requested_ranks('1,8', '8'), [1, 8])
        self.assertEqual(capsule.requested_ranks(None, '8', individual=True), [8])
        self.assertEqual(list(capsule.requested_ranks(None)), [1])
        for value in ('', '1,1', '0', '-1', '1,', '01', '1,9', '1,N', '2147483648'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                capsule.requested_ranks(value, '8')

    def test_shape_guards_and_custom_shapes(self):
        for count in (1, 2, 3, 4, 8, 16):
            for dimension, shape in capsule.DEFAULT_SHAPES.items():
                capsule.validate_shape(shape, dimension, count, 64)
        with self.assertRaisesRegex(ValueError, 'too small'):
            capsule.validate_shape(capsule.DEFAULT_SHAPES[4], 4, 17, 64)
        with self.assertRaisesRegex(ValueError, 'half-spectrum'):
            capsule.validate_shape([32, 40, 48], 3, 26, 64)
        capsule.validate_shape([32, 40, 64], 3, 32, 64)
        capsule.validate_shape([32, 40, 48, 64], 4, 32, 64)
        with self.assertRaisesRegex(ValueError, 'safety cap'):
            capsule.validate_shape([128] * 4, 4, 8, 64)

    def test_matrix_above_two_gpus_without_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'metadata').mkdir()
            (root / 'metadata/backend').write_text('cuda')
            caches = set()
            def solve(label, dim, shape, rank, transport, cache, *args, **kwargs):
                if cache not in caches:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text('cache fixture')
                    caches.add(cache)
                return True
            fake = mock.Mock()
            fake.preflight.return_value = True
            fake.solve.side_effect = solve
            fake.summary.return_value = 0
            argv = ['capsule.py', 'verify', '--ranks', '1,3,4', '--output', str(root / 'run')]
            with mock.patch.object(capsule, 'PREFIX', root), mock.patch.object(capsule, 'Runner', return_value=fake), \
                    mock.patch.object(sys, 'argv', argv), mock.patch.dict(os.environ, {'NGPU': '4'}):
                self.assertEqual(capsule.main(), 0)
            self.assertEqual(fake.preflight.call_count, 6)
            self.assertEqual(fake.solve.call_count, 48)
            self.assertEqual({call.args[3] for call in fake.solve.call_args_list}, {1, 3, 4})
            # Eight solves plus one preflight per transport and rank count.
            self.assertEqual(fake.preflight.call_count + fake.solve.call_count, 18 * 3)

    def test_invalid_shape_fails_before_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'metadata').mkdir()
            (root / 'metadata/backend').write_text('cuda')
            with mock.patch.object(capsule, 'PREFIX', root), mock.patch.object(capsule, 'Runner') as runner, \
                    mock.patch.object(sys, 'argv', ['capsule.py', 'verify', '--ranks', '32']), \
                    mock.patch.dict(os.environ, {'NGPU': '32'}), self.assertRaises(SystemExit) as failure:
                capsule.main()
            self.assertEqual(failure.exception.code, 2)
            runner.assert_not_called()

    def test_launcher_uses_explicit_context_and_separate_hip_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docker = root / 'fake-docker'
            docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$CAPSULE_COMMAND_LOG"\nprintf "[]\\n"\n')
            docker.chmod(0o755)
            log = root / 'commands'
            env = dict(os.environ, CAPSULE_DOCKER=str(docker), CAPSULE_COMMAND_LOG=str(log),
                       CAPSULE_DOCKER_CONTEXT='isolated-rootless', CAPSULE_HIP_VISIBLE_DEVICES='0',
                       NGPU='1', CAPSULE_HIP_DEVICES='/dev/dri/renderD128,/dev/dri/renderD129',
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


class LauncherTests(unittest.TestCase):
    def launch(self, backend='cuda', options=None, variables=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docker = root / 'fake-docker'
            docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$CAPSULE_COMMAND_LOG"\nprintf "[]\\n"\n')
            docker.chmod(0o755)
            smi = root / 'nvidia-smi'
            smi.write_text('#!/bin/sh\ni=0; while [ "$i" -lt "${CAPSULE_TEST_GPU_COUNT:-8}" ]; '
                           'do printf "%s\\n" "$i"; i=$((i+1)); done\n')
            smi.chmod(0o755)
            log = root / 'commands'
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith('CAPSULE_') and key != 'NGPU'}
            env.update(CAPSULE_DOCKER=str(docker), CAPSULE_COMMAND_LOG=str(log))
            env['PATH'] = str(root) + os.pathsep + env.get('PATH', '')
            env.update(variables or {})
            result = subprocess.run(['bash', str(ROOT / 'Docker_config/poisson/run.sh'), backend,
                                     'capsule:test', str(root / 'results'), *(options or [])],
                                    env=env, text=True, capture_output=True)
            return result, log.read_text().splitlines() if log.exists() else []

    def test_default_uses_one_cuda_gpu_and_one_rank(self):
        result, command = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('nvidia.com/gpu=0', command)
        self.assertNotIn('nvidia.com/gpu=1', command)
        self.assertEqual(command[-3:], ['verify', '--ranks', '1'])

    def test_two_gpus_keep_the_complete_matrix(self):
        result, command = self.launch(variables={'NGPU': '2'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('nvidia.com/gpu=0', command)
        self.assertIn('nvidia.com/gpu=1', command)
        self.assertEqual(command[-3:], ['verify', '--ranks', '1,2'])

    def test_general_gpu_counts_and_endpoint_matrix(self):
        for count in (3, 4, 8):
            with self.subTest(count=count):
                result, command = self.launch(variables={'NGPU': str(count)})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f'NGPU={count}', command)
                self.assertEqual(command[-1], ','.join(map(str, range(1, count + 1))))
                for index in range(count):
                    self.assertIn(f'nvidia.com/gpu={index}', command)
        result, command = self.launch(variables={'NGPU': '8'}, options=['verify', '--ranks=1,8'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(command[-1], '--ranks=1,8')

    def test_insufficient_cuda_devices_fail_before_docker(self):
        result, command = self.launch(variables={'NGPU': '4', 'CAPSULE_TEST_GPU_COUNT': '2'})
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('exceeds the 2 NVIDIA GPUs', result.stderr)
        self.assertEqual(command, [])

    def test_explicit_ranks_and_individual_solve(self):
        result, command = self.launch(options=['verify', '--ranks=2'], variables={'NGPU': '2'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(command[-2:], ['verify', '--ranks=2'])
        for ngpu in ('1', '2'):
            with self.subTest(ngpu=ngpu):
                result, command = self.launch(options=['poisson4d', '--transport', 'device-aware'],
                                              variables={'NGPU': ngpu})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(command[-2:], ['--ranks', ngpu])

    def test_invalid_gpu_counts_fail_before_docker(self):
        for value in ('', '0', '-1', '01', 'abc', '1+1', '2147483648', '999999999999999999999'):
            with self.subTest(value=value):
                result, command = self.launch(variables={'NGPU': value})
                self.assertEqual(result.returncode, 2)
                self.assertIn('NGPU must be a positive', result.stderr)
                self.assertEqual(command, [])

    def test_invalid_rank_requests_fail_before_docker(self):
        for options in (['verify', '--ranks', '2'], ['verify', '--ranks=1,2'],
                        ['verify', '--ranks'], ['verify', '--ranks='],
                        ['verify', '--ranks=1,1'], ['verify', '--ranks=3'],
                        ['verify', '--ranks=1', '--ranks', '1']):
            with self.subTest(options=options):
                result, command = self.launch(options=options)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(command, [])
        result, command = self.launch(options=['poisson3d', '--ranks=1,2'], variables={'NGPU': '2'})
        self.assertEqual(result.returncode, 2)
        self.assertEqual(command, [])

    def test_cuda_selection_must_match_gpu_count(self):
        result, command = self.launch(variables={'NGPU': '1', 'CAPSULE_CUDA_DEVICES': 'GPU-other'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('nvidia.com/gpu=GPU-other', command)
        for value in ('', '0,1', 'all', '0,', ',0', '0,,1'):
            with self.subTest(value=value):
                result, command = self.launch(variables={'CAPSULE_CUDA_DEVICES': value})
                self.assertEqual(result.returncode, 2)
                self.assertEqual(command, [])
        result, command = self.launch(variables={'NGPU': '2', 'CAPSULE_CUDA_DEVICES': '0,0'})
        self.assertEqual(result.returncode, 2)
        self.assertEqual(command, [])

    def test_hip_single_gpu_preserves_available_render_nodes(self):
        for nodes in ('/dev/dri/renderD128', '/dev/dri/renderD128,/dev/dri/renderD129'):
            with self.subTest(nodes=nodes):
                result, command = self.launch('hip', variables={'NGPU': '1', 'CAPSULE_HIP_DEVICES': nodes})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('ROCR_VISIBLE_DEVICES=0', command)
                self.assertEqual(command[-3:], ['verify', '--ranks', '1'])
                for node in nodes.split(','):
                    self.assertIn(node, command)
                if ',' not in nodes:
                    self.assertNotIn('/dev/dri/renderD129', command)

    def test_hip_two_gpu_mask_and_insufficient_nodes(self):
        variables = {'NGPU': '2', 'CAPSULE_HIP_DEVICES': '/dev/dri/renderD128,/dev/dri/renderD129'}
        result, command = self.launch('hip', variables=variables)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('ROCR_VISIBLE_DEVICES=0,1', command)
        self.assertEqual(command[-3:], ['verify', '--ranks', '1,2'])
        for update in ({'CAPSULE_HIP_VISIBLE_DEVICES': '0'},
                       {'CAPSULE_HIP_DEVICES': '/dev/dri/renderD128'},
                       {'CAPSULE_HIP_VISIBLE_DEVICES': '0,0'}):
            with self.subTest(update=update):
                result, command = self.launch('hip', variables=dict(variables, **update))
                self.assertEqual(result.returncode, 2)
                self.assertEqual(command, [])

    def test_hip_four_gpu_mapping(self):
        nodes = ','.join(f'/dev/dri/renderD{128 + i}' for i in range(4))
        result, command = self.launch('hip', variables={'NGPU': '4', 'CAPSULE_HIP_DEVICES': nodes})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('ROCR_VISIBLE_DEVICES=0,1,2,3', command)
        self.assertEqual(command[-1], '1,2,3,4')

    def test_cuda_inventory_rejects_malformed_results(self):
        with mock.patch.object(host_devices.subprocess, 'run', return_value=mock.Mock(stdout='0\n1\n2\n3\n')):
            self.assertEqual(host_devices.cuda_devices(), ['0', '1', '2', '3'])
        for output in ('', 'No devices found', '0\n0\n'):
            with mock.patch.object(host_devices.subprocess, 'run', return_value=mock.Mock(stdout=output)), \
                    self.assertRaises(ValueError):
                host_devices.cuda_devices()

    def test_amd_discovery_filters_other_vendors_and_missing_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dri, drm = root / 'dri', root / 'drm'
            dri.mkdir()
            for number, vendor in ((128, '0x1002'), (129, '0x10de'), (130, '0x1002')):
                device = drm / f'renderD{number}' / 'device'
                device.mkdir(parents=True)
                (device / 'vendor').write_text(vendor + '\n')
            (dri / 'renderD128').touch()
            (dri / 'renderD129').touch()
            (dri / 'renderD131').touch()
            self.assertEqual(host_devices.amd_render_devices(dri, drm), [str(dri / 'renderD128')])
            (dri / 'renderD130').touch()
            self.assertEqual(host_devices.amd_render_devices(dri, drm),
                             [str(dri / 'renderD128'), str(dri / 'renderD130')])
            self.assertEqual(host_devices.amd_render_devices(root / 'absent', drm), [])


if __name__ == '__main__':
    unittest.main()
