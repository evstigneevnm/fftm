#!/usr/bin/env python3
"""Export/load Docker archives in verified parts below the 2 GiB asset limit."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

PART_BYTES = 1900 * 1024 * 1024


def blocks(path):
    with path.open('rb') as stream:
        yield from iter(lambda: stream.read(1024 * 1024), b'')


def split_archive(archive, destination, part_bytes=PART_BYTES):
    digest, parts = hashlib.sha256(), []
    with archive.open('rb') as source:
        while True:
            head = source.read(min(part_bytes, 1024 * 1024))
            if not head:
                break
            name = f'{archive.name}.part{len(parts):03d}'
            checksum, size = hashlib.sha256(), 0
            with (destination / name).open('xb') as output:
                block = head
                while block:
                    output.write(block)
                    checksum.update(block)
                    digest.update(block)
                    size += len(block)
                    block = source.read(min(1024 * 1024, part_bytes - size)) if size < part_bytes else b''
            parts.append(dict(name=name, size=size, sha256=checksum.hexdigest()))
    return dict(schema=1, archive_sha256=digest.hexdigest(), parts=parts)


def verify(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 1 or not manifest.get('parts'):
        raise ValueError('invalid or empty archive manifest')
    whole, paths, names = hashlib.sha256(), [], set()
    for part in manifest['parts']:
        name = part['name']
        if Path(name).name != name or name in names or name in ('.', '..'):
            raise ValueError('invalid or duplicate part name')
        names.add(name)
        path = manifest_path.parent / name
        if path.is_symlink() or path.stat().st_size != part['size'] or not 0 < part['size'] <= PART_BYTES:
            raise ValueError(f'invalid part size or symlink: {name}')
        checksum = hashlib.sha256()
        for block in blocks(path):
            whole.update(block)
            checksum.update(block)
        if checksum.hexdigest() != part['sha256']:
            raise ValueError(f'part checksum mismatch: {name}')
        paths.append(path)
    if whole.hexdigest() != manifest['archive_sha256']:
        raise ValueError('combined archive checksum mismatch')
    return manifest, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docker', default='docker')
    parser.add_argument('--context', default='rootless')
    sub = parser.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export')
    export.add_argument('--image', required=True)
    export.add_argument('--output', type=Path, required=True)
    export.add_argument('--name', required=True, help='unique backend-specific release asset prefix')
    for name in ('verify', 'load'):
        command = sub.add_parser(name)
        command.add_argument('manifest', type=Path)
    args = parser.parse_args()
    docker = [args.docker, '--context', args.context]
    if args.command == 'export':
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.name):
            parser.error('--name must be a simple filename prefix')
        args.output.mkdir(parents=True, exist_ok=False)
        identity = json.loads(subprocess.check_output([*docker, 'image', 'inspect', args.image]))[0]
        archive = args.output / (args.name + '.tar.gz')
        with archive.open('xb') as stream:
            with gzip.GzipFile(filename='', mode='wb', fileobj=stream, compresslevel=1, mtime=0) as compressed:
                with subprocess.Popen([*docker, 'save', args.image], stdout=subprocess.PIPE) as process:
                    shutil.copyfileobj(process.stdout, compressed, 1024 * 1024)
                    if process.wait():
                        raise ValueError('docker save failed; partial archive retained for inspection')
        manifest = split_archive(archive, args.output)
        manifest.update(image=args.image, image_id=identity['Id'])
        path = args.output / (args.name + '.manifest.json')
        path.write_text(json.dumps(manifest, indent=2) + '\n')
        (args.output / (args.name + '.manifest.sha256')).write_text(
            hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.name + '\n')
        verify(path)
        print(f'Exported {len(manifest["parts"])} parts; upload *.manifest.*, *.part*, not the full .tar.gz')
    else:
        manifest, paths = verify(args.manifest)
        print('All parts and combined checksum verified.', flush=True)
        if args.command == 'load':
            with subprocess.Popen([*docker, 'load'], stdin=subprocess.PIPE) as process:
                for path in paths:
                    for block in blocks(path):
                        process.stdin.write(block)
                process.stdin.close()
                if process.wait():
                    raise ValueError('docker load failed')
            identity = json.loads(subprocess.check_output([*docker, 'image', 'inspect', manifest['image']]))[0]
            if identity['Id'] != manifest['image_id']:
                raise ValueError('loaded image identity does not match the export')
            print('Loaded ' + identity['Id'])
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print('ARCHIVE_ERROR: ' + str(error), file=sys.stderr)
        sys.exit(1)
