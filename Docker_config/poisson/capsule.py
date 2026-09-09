#!/usr/bin/env python3
"""Small, retained-log correctness capsule. Timings are not paper benchmarks."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

PREFIX = Path('/opt/fftm')
DEFAULT_SHAPES = {3: [32, 40, 48], 4: [16, 20, 24, 32]}


def positive_count(value):
    if not re.fullmatch(r'[1-9][0-9]{0,9}', value) or int(value) > 2147483647:
        raise ValueError('GPU/rank counts must be positive decimal integers within the MPI count range')
    return int(value)


def requested_ranks(value, ngpu=None, individual=False):
    count = positive_count(ngpu) if ngpu is not None else None
    if value is None:
        return [count or 1] if individual else range(1, (count or 1) + 1)
    ranks = [positive_count(part) for part in value.split(',')]
    if len(set(ranks)) != len(ranks):
        raise ValueError('--ranks must contain distinct counts')
    if count is not None and max(ranks) > count:
        raise ValueError('--ranks exceeds the requested NGPU')
    return ranks


def validate_shape(sizes, dimension, ranks, max_input_mib):
    if len(sizes) != dimension or min(sizes) < 8:
        raise ValueError(f'{dimension}D requires {dimension} axis sizes, each at least 8')
    # A sufficient bound for every candidate grid, without reproducing the C++ grid policy.
    stored = [*sizes[:-1], sizes[-1] // 2 + 1]
    if min(stored) < ranks:
        raise ValueError(f'{dimension}D shape {sizes} is too small for {ranks} ranks in this capsule: '
                         'every full axis and the stored half-spectrum must span at least that '
                         f'many entries; supply larger --sizes-{dimension}d (verify) or --sizes (solve)')
    input_bytes = 8 * math.prod(sizes)
    if input_bytes > max_input_mib * 1024 * 1024:
        raise ValueError(f'{dimension}D real input exceeds the {max_input_mib} MiB capsule safety cap; '
                         'reduce the shape or explicitly raise --max-input-mib after checking memory')


def result_fields(text, dimension):
    matches = re.findall(r'poisson_periodic_' + str(dimension) + r'd_autotuned: ([^\n]+)', text)
    if len(matches) != 1:
        raise ValueError('expected exactly one solver result')
    fields = dict(re.findall(r'(\w+)=(\S+)', matches[0]))
    error = float(fields['rel_l2'])
    if not math.isfinite(error) or not 0 <= error < 1.0e-10:
        raise ValueError('relative L2 error is not finite and below 1e-10')
    return fields


def mpi_environment(transport):
    env = os.environ.copy()
    # Do not inherit arbitrary tuning flags into a claimed reproduction matrix.
    for key in list(env):
        if key.startswith(('FFTM_', 'OMPI_MCA_', 'UCX_')):
            del env[key]
    env.update(FFTM_WRAP_PROCS_GPUS='0', FFTM_CPP_AUTOTUNE_MEASURE='1',
               FFTM_CPP_AUTOTUNE_WARMUP='1', FFTM_CPP_AUTOTUNE_TIMES='2',
               FFTM_CPP_AUTOTUNE_STRICT_DEVICE_IDENTITY='1',
               FFTM_CPP_AUTOTUNE_ACCEPTED_SPECTRAL_LAYOUTS_4D='public-yzwx,native-xzwy',
               OMP_NUM_THREADS='1')
    if os.geteuid() == 0:
        env.update(OMPI_ALLOW_RUN_AS_ROOT='1', OMPI_ALLOW_RUN_AS_ROOT_CONFIRM='1')
    if transport == 'device-aware':
        env.update(OMPI_MCA_pml='ucx', OMPI_MCA_pml_ucx_devices='any', OMPI_MCA_pml_ucx_tls='any')
        # A declared capsule setting is distinct from arbitrary inherited UCX tuning.
        if env.get('CAPSULE_UCX_TLS'):
            env['UCX_TLS'] = env['CAPSULE_UCX_TLS']
    else:
        env.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,sm,tcp')
    return env


class Runner:
    def __init__(self, output, backend, timeout, request=None):
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'status.jsonl').exists():
            raise ValueError('output already has a status.jsonl; choose a new output directory')
        self.backend, self.timeout, self.records = backend, timeout, []
        metadata = {p.name: p.read_text() for p in (PREFIX / 'metadata').glob('*.json')}
        metadata.update(backend=backend, hostname=os.uname().nodename, started=time.time(),
                        timeout_seconds=timeout, purpose='correctness, not performance')
        if request is not None:
            metadata['request'] = request
        (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')

    def execute(self, label, command, env, validate):
        record = dict(label=label, command=command, started=time.time())
        log = self.output / (label + '.log')
        (self.output / (label + '.command.json')).write_text(json.dumps(
            dict(argv=command, environment={k: v for k, v in env.items()
                                          if k.startswith(('FFTM_', 'OMPI_', 'UCX_', 'OMP_')) or
                                          k in ('NGPU', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')}), indent=2) + '\n')
        print('RUN ' + label + ': ' + shlex.join(command), flush=True)
        try:
            with log.open('w') as stream:
                stream.write('COMMAND ' + shlex.join(command) + '\n')
                stream.flush()
                with subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                      start_new_session=True) as process:
                    try:
                        record['returncode'] = process.wait(timeout=self.timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        record['returncode'] = 124
                        raise ValueError('timeout; MPI process group terminated')
            record['result'] = validate(record['returncode'], log.read_text(errors='replace'))
            record['passed'] = True
        except (OSError, ValueError, KeyError) as error:
            record.update(passed=False, error=str(error))
        record['elapsed_seconds'] = time.time() - record['started']
        self.records.append(record)
        with (self.output / 'status.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(('PASS ' if record['passed'] else 'FAIL ') + label, flush=True)
        return record['passed']

    def mpi(self, ranks, executable, args):
        return ['/opt/mpi/bin/mpiexec', '--bind-to', 'none', '-n', str(ranks),
                str(PREFIX / 'bin' / executable), *map(str, args)]

    def preflight(self, ranks, transport):
        def validate(rc, text):
            if rc != 0 or text.count('CAPSULE_PROBE_PASS') != 1:
                raise ValueError('GPU/MPI buffer preflight failed; inspect log')
            devices = [json.loads(line.split('CAPSULE_DEVICE ', 1)[1])
                       for line in text.splitlines() if line.startswith('CAPSULE_DEVICE ')]
            if len(devices) != ranks or len({v['pci'] for v in devices}) != ranks:
                raise ValueError('missing or duplicate rank-to-GPU mapping')
            return devices
        return self.execute(f'preflight-r{ranks}-{transport}',
                            self.mpi(ranks, 'capsule_probe.bin', [transport]),
                            mpi_environment(transport), validate)

    def solve(self, label, dimension, sizes, ranks, transport, cache, source=None,
              layouts='public-yzwx,native-xzwy', negative=False, before=None):
        suffix = '_hip' if self.backend == 'hip' else ''
        suffix += '_nca' if transport == 'host-staged' else ''
        env = mpi_environment(transport)
        env['FFTM_CPP_AUTOTUNE_ACCEPTED_SPECTRAL_LAYOUTS_4D'] = layouts
        def validate(rc, text):
            if negative:
                cache_error = re.search(
                    r'FFTM (?:4D autotune cache is invalid or mismatched|autotune config[^\n]*does not match)', text)
                if rc == 0 or cache_error is None:
                    raise ValueError('expected explicit cache mismatch rejection, not another failure')
                if before != cache.read_bytes():
                    raise ValueError('rejected cache was modified')
                return dict(expected_rejection=True)
            if rc != 0:
                raise ValueError(f'solver exit code {rc}')
            fields = result_fields(text, dimension)
            if dimension == 4 and fields.get('layout') not in layouts.split(','):
                raise ValueError('selected 4D layout is outside the accepted layouts')
            expected = dict(mpi=str(ranks), backend=self.backend,
                            device_aware_mpi=str(int(transport == 'device-aware')),
                            size='x'.join(map(str, sizes)))
            if source is not None:
                expected['source'] = source
            for key, value in expected.items():
                if fields.get(key) != value:
                    raise ValueError(f'{key}: expected {value}, got {fields.get(key)}')
            if not cache.is_file() or not cache.stat().st_size:
                raise ValueError('missing cache')
            if before is not None and before != cache.read_bytes():
                raise ValueError('cache changed during reuse')
            return fields
        return self.execute(label, self.mpi(ranks, f'poisson_periodic_{dimension}d_autotuned{suffix}.bin',
                                           [*sizes, cache, 2, 1]), env, validate)

    def summary(self):
        report = dict(passed=all(r['passed'] for r in self.records), cases=len(self.records),
                      failures=[r['label'] for r in self.records if not r['passed']])
        (self.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
        return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('info', 'preflight', 'verify', 'poisson3d', 'poisson4d'))
    parser.add_argument('--ranks', help='distinct counts; default 1..NGPU for a matrix, NGPU for a solve')
    parser.add_argument('--transport', choices=('both', 'device-aware', 'host-staged'), default='both')
    parser.add_argument('--output', type=Path, default=Path('/data/run'))
    parser.add_argument('--timeout', type=int, default=300, help='seconds per MPI invocation')
    parser.add_argument('--sizes', type=int, nargs='+')
    parser.add_argument('--sizes-3d', type=int, nargs=3, help='verification shape, default 32 40 48')
    parser.add_argument('--sizes-4d', type=int, nargs=4, help='verification shape, default 16 20 24 32')
    parser.add_argument('--max-input-mib', type=int, default=64,
                        help='safety cap on global real input per problem, not a total workspace estimate')
    parser.add_argument('--cache', type=Path, help='individual solve only; missing files are created')
    parser.add_argument('--layouts', default='public-yzwx,native-xzwy',
                        choices=('public-yzwx', 'native-xzwy', 'public-yzwx,native-xzwy'))
    args = parser.parse_args()
    backend = (PREFIX / 'metadata/backend').read_text().strip()
    if args.command == 'info':
        for name in ('backend', 'provenance.json', 'build-config.txt', 'compiler.txt', 'ucx-version.txt'):
            print(f'=== {name} ===\n' + (PREFIX / 'metadata' / name).read_text())
        return 0
    transports = ['device-aware', 'host-staged'] if args.transport == 'both' else [args.transport]
    individual = args.command.startswith('poisson')
    dimension = 3 if args.command == 'poisson3d' else 4
    try:
        ranks = requested_ranks(args.ranks, os.environ.get('NGPU'), individual)
    except ValueError as error:
        parser.error(str(error))
    if args.timeout < 1 or args.max_input_mib < 1:
        parser.error('--timeout and --max-input-mib must be positive')
    if individual and (len(ranks) != 1 or len(transports) != 1 or args.cache is None
                       or args.sizes is None or len(args.sizes) != dimension or min(args.sizes) < 8):
        parser.error('individual solves require one rank count, one transport, --cache, and all axis sizes >=8')
    if not individual and (args.sizes is not None or args.cache is not None):
        parser.error('--sizes and --cache are for individual solves; use --sizes-3d/--sizes-4d for verify')
    if args.command != 'verify' and (args.sizes_3d is not None or args.sizes_4d is not None):
        parser.error('--sizes-3d and --sizes-4d require verify')
    shapes = {3: args.sizes_3d or DEFAULT_SHAPES[3], 4: args.sizes_4d or DEFAULT_SHAPES[4]}
    try:
        if individual:
            validate_shape(args.sizes, dimension, max(ranks), args.max_input_mib)
        elif args.command == 'verify':
            for dim, shape in shapes.items():
                validate_shape(shape, dim, max(ranks), args.max_input_mib)
                validate_shape([shape[0] + 8, *shape[1:]], dim, max(ranks), args.max_input_mib)
    except ValueError as error:
        parser.error(str(error))
    runner = Runner(args.output, backend, args.timeout, request=dict(
        command=args.command, ngpu=os.environ.get('NGPU'), ranks=list(ranks),
        transports=transports, shapes={dimension: args.sizes} if individual else shapes,
        max_input_mib=args.max_input_mib))
    for rank in ranks:
        for transport in transports:
            if not runner.preflight(rank, transport):
                continue
            if args.command == 'preflight':
                continue
            if individual:
                args.cache.parent.mkdir(parents=True, exist_ok=True)
                runner.solve('solve', dimension, args.sizes, rank, transport, args.cache, layouts=args.layouts)
                continue
            for dim, shape in shapes.items():
                label = f'd{dim}-r{rank}-{transport}'
                cache = args.output / (label + '.env')
                if cache.exists():
                    raise ValueError(f'refusing to overwrite {cache}')
                if not runner.solve(label + '-create', dim, shape, rank, transport, cache, 'measured'):
                    continue
                before = cache.read_bytes()
                runner.solve(label + '-reuse', dim, shape, rank, transport, cache, 'cache', before=before)
                wrong = list(shape)
                wrong[0] += 8
                runner.solve(label + '-size-mismatch', dim, wrong, rank, transport, cache,
                             negative=True, before=before)
            # Exercise both physical 4D contracts even if tuning always picks one.
            for layout in ('public-yzwx', 'native-xzwy'):
                label = f'd4-r{rank}-{transport}-{layout}'
                runner.solve(label, 4, shapes[4], rank, transport,
                             args.output / (label + '.env'), 'measured', layouts=layout)
    return runner.summary()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print('CAPSULE_ERROR: ' + str(error), file=sys.stderr)
        sys.exit(1)
