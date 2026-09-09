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
architectures = module('check_architectures')
builder = module('build')


class CapsuleTests(unittest.TestCase):
    def test_build_context_filters_backend_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / 'repo'
            here = repo / 'Docker_config/poisson'
            here.mkdir(parents=True)
            (repo / 'README.md').write_text('source fixture')
            cache = root / 'cache'
            cache.mkdir()
            specs = {}
            for name, backends in (('common.tar.gz', ['cuda', 'hip']), ('rocfft.tar.gz', ['hip'])):
                path = cache / name
                path.write_bytes(name.encode())
                specs[name] = dict(url='https://invalid.example/' + name,
                                   sha256=builder.sha256(path), backends=backends)
            (here / 'dependencies.json').write_text(json.dumps(specs))
            def fake_git(*args, **kwargs):
                if args[0] == 'rev-parse':
                    return b'a' * 40
                if args[0] == 'ls-files' and 'cwd' not in kwargs and '--others' not in args:
                    return b'README.md\0'
                return b''
            with mock.patch.object(builder, 'ROOT', repo), mock.patch.object(builder, 'HERE', here), \
                    mock.patch.object(builder, 'git', side_effect=fake_git):
                for backend in ('cuda', 'hip'):
                    context = builder.prepare(root / backend, cache, backend)
                    self.assertTrue((context / 'deps/common.tar.gz').exists())
                    self.assertEqual((context / 'deps/rocfft.tar.gz').exists(), backend == 'hip')

    def test_architecture_listings_reject_missing_targets(self):
        self.assertEqual(architectures.embedded_targets(
            'ELF file 1: example.1.sm_60.cubin\nELF file 2: example.2.sm_120.cubin', 'cuda'),
            ['120', '60'])
        self.assertEqual(architectures.embedded_targets(
            'amdgcn-amd-amdhsa--gfx90a:xnack-\namdgcn-amd-amdhsa--gfx1102', 'hip'),
            ['gfx1102', 'gfx90a'])
        architectures.check_targets(['gfx90a'], ['gfx90a', 'gfx1102'], 'probe')
        with self.assertRaisesRegex(ValueError, 'missing.*gfx906'):
            architectures.check_targets(['gfx906', 'gfx1102'], ['gfx1102'], 'probe')

    def test_hip_runtime_only_exception_is_limited_to_probe(self):
        listing = json.dumps([dict(Sections=[dict(Section=dict(Name=dict(Name='.text')))])])
        with mock.patch.object(architectures.subprocess, 'check_output', return_value=listing):
            self.assertIsNone(architectures.hip_listing(Path('/bin/capsule_probe.bin')))
            for name in ('poisson_periodic_4d_autotuned_hip.bin', 'librocfft.so'):
                with self.assertRaisesRegex(ValueError, 'missing HIP code bundle'):
                    architectures.hip_listing(Path('/bin') / name)

    def test_hip_bundle_is_inspected_when_present(self):
        sections = json.dumps([dict(Sections=[dict(Section=dict(Name=dict(Name='.hip_fatbin')))])])
        bundle = 'hip-amdgcn-amd-amdhsa--gfx1102\n'
        with mock.patch.object(architectures.subprocess, 'check_output', side_effect=[sections, bundle]), \
                mock.patch.object(architectures.subprocess, 'run') as objcopy:
            self.assertEqual(architectures.hip_listing(Path('/bin/capsule_probe.bin')), bundle)
            self.assertIn('--dump-section', objcopy.call_args.args[0])

    def test_rocfft_runtime_configuration_requires_all_targets(self):
        config = 'ROCFFT_RUNTIME_COMPILE_DEFAULT:BOOL=ON\nGPU_TARGETS:STRING=gfx906;gfx1102\n'
        self.assertEqual(architectures.rocfft_rtc_config(config, ['gfx906', 'gfx1102']),
                         ['gfx906', 'gfx1102'])
        with self.assertRaisesRegex(ValueError, 'runtime compilation'):
            architectures.rocfft_rtc_config(config.replace('=ON', '=OFF'), ['gfx906'])
        with self.assertRaisesRegex(ValueError, 'missing.*gfx1200'):
            architectures.rocfft_rtc_config(config, ['gfx1200'])

    def test_capsule_profiles_cover_requested_architectures(self):
        expected = {'cuda': '60 61 70 75 80 86 89 90 100 103 120',
                    'hip': 'gfx906 gfx908 gfx90a gfx942 gfx1030 gfx1031 gfx1032 '
                           'gfx1100 gfx1101 gfx1102 gfx1200 gfx1201'}
        for backend, values in expected.items():
            profile = f'Docker_config/poisson/config_{backend}.inc'
            output = subprocess.check_output(['make', '-s', '-C', str(ROOT / 'examples/poisson'),
                                              f'CONFIG_FILE={profile}', 'print-config'], text=True)
            config = dict(line.split('=', 1) for line in output.splitlines() if '=' in line)
            self.assertEqual(config[backend.upper() + '_ARCH_LIST'], values)
            if backend == 'cuda':
                self.assertIn('arch=compute_60,code=compute_60', config['NVCCFLAGS'])

    def test_probe_and_solvers_share_overridable_architecture_flags(self):
        for backend, targets in (('cuda', ['70', '90']), ('hip', ['gfx90a', 'gfx1102'])):
            arguments = [f'CONFIG_FILE=Docker_config/poisson/config_{backend}.inc',
                         backend.upper() + '_ARCH_LIST=' + ' '.join(targets), 'all']
            probe = subprocess.check_output(['make', '-n', '-f',
                                             str(ROOT / 'Docker_config/poisson/Makefile.probe'),
                                             *arguments], cwd=ROOT, text=True)
            solvers = subprocess.check_output(['make', '-n', '-C', str(ROOT / 'examples/poisson'),
                                               *arguments], text=True)
            for target in targets:
                flag = (f'arch=compute_{target},code=sm_{target}' if backend == 'cuda'
                        else f'--offload-arch={target}')
                self.assertIn(flag, probe)
                self.assertIn(flag, solvers)
            self.assertNotIn('code=sm_86', probe)
            if backend == 'hip':
                self.assertNotIn('--offload-arch=gfx906', probe)

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
