#!/usr/bin/env python3
"""Discover host GPU devices without initializing a compute context."""
import csv
from pathlib import Path
import subprocess
import sys


def amd_render_devices(dri=Path('/dev/dri'), drm=Path('/sys/class/drm')):
    devices = []
    for device in sorted(dri.glob('renderD*')):
        try:
            vendor = (drm / device.name / 'device/vendor').read_text().strip().lower()
        except OSError:
            continue
        if device.exists() and vendor == '0x1002':
            devices.append(str(device))
    return devices


def cuda_devices():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'],
                            check=True, text=True, capture_output=True, timeout=15)
    devices = [row[0].strip() for row in csv.reader(result.stdout.splitlines()) if row]
    if not devices or any(not value.isascii() or not value.isdecimal() for value in devices):
        raise ValueError('nvidia-smi did not report a valid GPU index list')
    if len(set(devices)) != len(devices):
        raise ValueError('nvidia-smi reported duplicate GPU indices')
    return devices


if __name__ == '__main__':
    try:
        backend = sys.argv[1] if len(sys.argv) == 2 else 'hip'
        if backend == 'cuda':
            devices = cuda_devices()
        elif backend == 'hip':
            devices = amd_render_devices()
            if not devices:
                raise ValueError('No AMD render nodes found. Check /dev/dri and /sys/class/drm, '
                                 'or set CAPSULE_HIP_DEVICES explicitly.')
        else:
            raise ValueError('Backend must be cuda or hip')
        print(','.join(devices))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print('GPU discovery failed: ' + str(error), file=sys.stderr)
        sys.exit(2)
