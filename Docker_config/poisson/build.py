#!/usr/bin/env python3
"""Create an allowlisted, checksummed source context and build one capsule."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', '-C', str(cwd), *args])


def prepare(output, cache):
    context = output / 'context'
    context.mkdir(parents=True)
    scopes = ['source', 'build_configs', 'examples/poisson', 'examples/tests',
              'Docker_config/poisson', 'README.md', 'LICENSE', '.gitmodules']
    paths = set(git('ls-files', '-z', '--', *scopes).decode().strip('\0').split('\0'))
    paths.update(git('ls-files', '--others', '--exclude-standard', '-z', '--',
                     'Docker_config/poisson').decode().strip('\0').split('\0'))
    submodule = ROOT / 'source/contrib/scfd'
    paths.update('source/contrib/scfd/' + p for p in
                 git('ls-files', '-z', cwd=submodule).decode().strip('\0').split('\0') if p)
    sums = []
    for name in sorted(paths):
        source = ROOT / name
        if not name or source.is_dir():
            continue
        if source.is_symlink():
            raise ValueError(f'unexpected source symlink: {name}')
        target = context / 'src' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        sums.append(f'{sha256(target)}  {name}\n')
    manifest = ''.join(sums)
    (context / 'src/source.sha256').write_text(manifest)
    provenance = dict(git_commit=git('rev-parse', 'HEAD').decode().strip(),
                      scfd_commit=git('rev-parse', 'HEAD', cwd=submodule).decode().strip(),
                      source_manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(),
                      source_file_count=len(sums),
                      worktree_status=git('status', '--porcelain', '--', *scopes).decode(),
                      scfd_worktree_status=git('status', '--porcelain', cwd=submodule).decode())
    (context / 'src/provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    deps = json.loads((HERE / 'dependencies.json').read_text())
    cache.mkdir(parents=True, exist_ok=True)
    (context / 'deps').mkdir()
    checksums = []
    for name, spec in deps.items():
        path = cache / name
        if not path.exists():
            partial = path.with_suffix(path.suffix + '.partial')
            subprocess.run(['curl', '-fL', '--connect-timeout', '20', '--max-time', '600',
                            '--retry', '2', spec['url'], '-o', str(partial)], check=True)
            if sha256(partial) != spec['sha256']:
                raise ValueError(f'download checksum mismatch: {name}')
            partial.rename(path)
        if sha256(path) != spec['sha256']:
            raise ValueError(f'cached dependency checksum mismatch: {name}; remove only this file and retry')
        shutil.copy2(path, context / 'deps' / name)
        checksums.append(f"{spec['sha256']}  {name}\n")
    (context / 'deps/SHA256SUMS').write_text(''.join(checksums))
    return context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend', choices=('cuda', 'hip'))
    parser.add_argument('--tag', required=True)
    parser.add_argument('--output', type=Path, required=True, help='new build-log/context directory')
    parser.add_argument('--context', default='rootless', help='Docker context; never changed globally')
    parser.add_argument('--docker', default='docker')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--cache', type=Path, default=ROOT / 'build/poisson_capsule_deps')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or args.jobs < 1:
        parser.error('--output must be new and --jobs positive')
    output = args.output.resolve()
    context = prepare(output, args.cache.resolve())
    docker = [args.docker, '--context', args.context]
    command = [*docker, 'build', '--progress=plain', '--build-arg', f'BUILD_JOBS={args.jobs}',
               '--label', 'org.opencontainers.image.title=FFTM Poisson reproducibility capsule',
               '-f', str(context / 'src/Docker_config/poisson' / ('Dockerfile.' + args.backend)),
               '-t', args.tag, str(context)]
    (output / 'build-command.json').write_text(json.dumps(command, indent=2) + '\n')
    print('Context: ' + str(context), flush=True)
    if args.prepare_only:
        return 0
    print('Build log: ' + str(output / 'build.log'), flush=True)
    with (output / 'build.log').open('w') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print('\n'.join((output / 'build.log').read_text(errors='replace').splitlines()[-45:]), file=sys.stderr)
        return result.returncode
    image = subprocess.check_output([*docker, 'image', 'inspect', args.tag])
    (output / 'image-inspect.json').write_bytes(image)
    print('Built ' + args.tag + ': ' + json.loads(image)[0]['Id'])
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
