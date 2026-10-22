"""Small, self-contained run manifests for the paleoecology workflows."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
from typing import Mapping

import numpy as np


def _canonical(value):
    if isinstance(value, np.ndarray):
        return {'array': _canonical(value.tolist()), 'dtype': str(value.dtype), 'shape': list(value.shape)}
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {'nonfinite': str(value)}
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f'Cannot create a reproducible manifest for {type(value).__name__}.')


def content_hash(value) -> str:
    """Hash input structure, values, missingness, array dtype and dimensions."""
    encoded = json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def file_fingerprint(path) -> dict:
    source = Path(path).resolve(strict=True)
    if not source.is_file():
        raise ValueError(f'Expected a source file: {source}')
    return {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}


def build_run_manifest(inputs, parameters, outputs, *, source_files=()) -> dict:
    """Describe an actual run; preserve source IDs and raw fields in inputs.

    File fingerprints are added only for explicitly supplied files. Hashes
    prove identity, not accuracy of extraction or ecological interpretation.
    """
    versions = {'python': platform.python_version()}
    for package in ('numpy', 'pandas', 'scipy', 'statsmodels'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    scripts = sorted(Path(__file__).parent.glob('*.py'))
    return {
        'schema_version': '1.0',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'input_sha256': content_hash(inputs),
        'parameters': _canonical(parameters),
        'output_sha256': content_hash(outputs),
        'source_files': [file_fingerprint(path) for path in source_files],
        'code_files': [file_fingerprint(path) for path in scripts],
        'software_versions': versions,
        'verification': 'computational_manifest; source verification remains a separate judgment',
    }
