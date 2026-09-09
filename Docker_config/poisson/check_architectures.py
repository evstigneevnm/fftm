#!/usr/bin/env python3
"""Verify embedded executable targets; this does not establish hardware support."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def embedded_targets(text, backend):
    pattern = r'\bsm_([0-9]+)\b' if backend == 'cuda' else r'\b(gfx[0-9a-f]+)\b'
    return sorted(set(re.findall(pattern, text)))


def check_targets(required, found, name):
    missing = sorted(set(required) - set(found))
    if missing:
        raise ValueError(f'{name}: missing executable GPU targets: {", ".join(missing)}')


def hip_listing(binary, runtime_only=False):
    sections = json.loads(subprocess.check_output(
        ['/opt/rocm/llvm/bin/llvm-readobj', '--elf-output-style=JSON', '--sections', str(binary)],
        text=True))[0]['Sections']
    names = {section['Section']['Name']['Name'] for section in sections}
    if '.hip_fatbin' not in names:
        if binary.name == 'capsule_probe.bin' or runtime_only:
            # This probe calls runtime allocation/copy APIs without launching its own kernels.
            return None
        raise ValueError(f'{binary.name}: missing HIP code bundle')
    # LLVM reads the structured offload bundle, including compressed code objects.
    with tempfile.TemporaryDirectory() as directory:
        bundle = Path(directory) / 'bundle'
        subprocess.run(['/opt/rocm/llvm/bin/llvm-objcopy', '--dump-section',
                        '.hip_fatbin=' + str(bundle), str(binary), directory + '/object'], check=True)
        return subprocess.check_output(['/opt/rocm/llvm/bin/clang-offload-bundler', '--type=o',
                                        '--list', '--input=' + str(bundle)], text=True,
                                       stderr=subprocess.STDOUT)


def rocfft_rtc_config(cache_text, required):
    config = {}
    for line in cache_text.splitlines():
        if not line.startswith(('#', '//')) and '=' in line:
            key, value = line.split('=', 1)
            config[key.split(':', 1)[0]] = value
    if config.get('ROCFFT_RUNTIME_COMPILE_DEFAULT') != 'ON':
        raise ValueError('rocFFT must be rebuilt with runtime compilation enabled by default')
    targets = [target.split(':', 1)[0] for target in config.get('GPU_TARGETS', '').split(';')]
    check_targets(required, targets, 'rocFFT build configuration')
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend', choices=('cuda', 'hip'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--bin-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = dict(line.split('=', 1) for line in args.config.read_text().splitlines() if '=' in line)
    required = config['CUDA_ARCH_LIST' if args.backend == 'cuda' else 'HIP_ARCH_LIST'].split()
    if not required or len(set(required)) != len(required):
        raise ValueError('expected a nonempty list of distinct GPU targets')
    binaries = sorted(args.bin_dir.glob('*.bin'))
    if len(binaries) != 5:
        raise ValueError(f'expected four Poisson executables and one probe, found {len(binaries)}')
    report = dict(backend=args.backend, requested=required, binaries={},
                  qualification='Compiled targets only; not a hardware-validation claim')
    for binary in binaries:
        listing = (subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '--list-elf', str(binary)],
                                            text=True, stderr=subprocess.STDOUT)
                   if args.backend == 'cuda' else hip_listing(binary))
        if listing is None:
            report['binaries'][binary.name] = dict(
                targets=[], runtime_only=True,
                qualification='No embedded HIP kernels; uses vendor allocation/copy APIs')
            continue
        (args.output.parent / (binary.name + '.architectures.txt')).write_text(listing)
        found = embedded_targets(listing, args.backend)
        check_targets(required, found, binary.name)
        record = dict(targets=found)
        if args.backend == 'cuda':
            ptx = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '--list-ptx', str(binary)],
                                          text=True, stderr=subprocess.STDOUT)
            check_targets(['60'], embedded_targets(ptx, 'cuda'), binary.name + ' PTX fallback')
            record['ptx_targets'] = embedded_targets(ptx, 'cuda')
            (args.output.parent / (binary.name + '.ptx.txt')).write_text(ptx)
        report['binaries'][binary.name] = record
    if args.backend == 'hip':
        targets = rocfft_rtc_config(
            (args.output.parent / 'rocfft-CMakeCache.txt').read_text(), required)
        listing = hip_listing(Path('/opt/rocm/lib/librocfft.so'), runtime_only=True)
        report['rocfft'] = dict(configured_targets=targets, runtime_compile_default=True)
        if listing is None:
            report['rocfft']['runtime_only'] = True
        else:
            (args.output.parent / 'rocfft.architectures.txt').write_text(listing)
            found = embedded_targets(listing, 'hip')
            check_targets(required, found, 'librocfft.so')
            report['rocfft']['embedded_targets'] = found
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Checked {len(binaries)} {args.backend} binaries for {len(required)} requested GPU targets')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        print(f'ARCHITECTURE_ERROR: {error}', file=sys.stderr)
        sys.exit(1)
