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
    def __init__(self, output, backend, timeout):
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'status.jsonl').exists():
            raise ValueError('output already has a status.jsonl; choose a new output directory')
        self.backend, self.timeout, self.records = backend, timeout, []
        metadata = {p.name: p.read_text() for p in (PREFIX / 'metadata').glob('*.json')}
        metadata.update(backend=backend, hostname=os.uname().nodename, started=time.time(),
                        timeout_seconds=timeout, purpose='correctness, not performance')
        (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')

    def execute(self, label, command, env, validate):
        record = dict(label=label, command=command, started=time.time())
        log = self.output / (label + '.log')
        (self.output / (label + '.command.json')).write_text(json.dumps(
            dict(argv=command, environment={k: v for k, v in env.items()
                                          if k.startswith(('FFTM_', 'OMPI_', 'UCX_', 'OMP_'))}), indent=2) + '\n')
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
    parser.add_argument('--ranks', default='1,2', help='verify list, or a single count for an individual solve')
    parser.add_argument('--transport', choices=('both', 'device-aware', 'host-staged'), default='both')
    parser.add_argument('--output', type=Path, default=Path('/data/run'))
    parser.add_argument('--timeout', type=int, default=300, help='seconds per MPI invocation')
    parser.add_argument('--sizes', type=int, nargs='+')
    parser.add_argument('--cache', type=Path, help='individual solve only; missing files are created')
    parser.add_argument('--layouts', default='public-yzwx,native-xzwy',
                        choices=('public-yzwx', 'native-xzwy', 'public-yzwx,native-xzwy'))
    args = parser.parse_args()
    backend = (PREFIX / 'metadata/backend').read_text().strip()
    if args.command == 'info':
        for name in ('backend', 'provenance.json', 'build-config.txt', 'compiler.txt', 'ucx-version.txt'):
            print(f'=== {name} ===\n' + (PREFIX / 'metadata' / name).read_text())
        return 0
    ranks = list(dict.fromkeys(int(r) for r in args.ranks.split(',')))
    if not ranks or min(ranks) < 1 or max(ranks) > 2 or args.timeout < 1:
        parser.error('this small single-node capsule supports one or two ranks and a positive timeout')
    transports = ['device-aware', 'host-staged'] if args.transport == 'both' else [args.transport]
    individual = args.command.startswith('poisson')
    dimension = 3 if args.command == 'poisson3d' else 4
    if individual and (len(ranks) != 1 or len(transports) != 1 or args.cache is None
                       or args.sizes is None or len(args.sizes) != dimension or min(args.sizes) < 8):
        parser.error('individual solves require one rank count, one transport, --cache, and all axis sizes >=8')
    runner = Runner(args.output, backend, args.timeout)
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
            for dim, shape in ((3, [32, 40, 48]), (4, [16, 20, 24, 32])):
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
                runner.solve(label, 4, [16, 20, 24, 32], rank, transport,
                             args.output / (label + '.env'), 'measured', layouts=layout)
    return runner.summary()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print('CAPSULE_ERROR: ' + str(error), file=sys.stderr)
        sys.exit(1)
