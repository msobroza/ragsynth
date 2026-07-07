"""Artifact persistence with a sha256 manifest.

Fitted statistical objects (reference partition, movMF demand map) are
"versioned frozen artifacts per benchmark epoch" (SPEC §7): every save
records the artifact's sha256 in ``manifest.json`` so configs can
reference artifacts by relative path + hash (SPEC §13).
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


def sha256_hex(data: bytes) -> str:
    """Return the hex sha256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> str:
    """Serialize ``obj`` to canonical JSON (sorted keys, no whitespace drift).

    Used for config hashing: two configs with the same content but
    different key order produce identical hashes.

    Args:
        obj: Any JSON-serializable object.

    Returns:
        A deterministic JSON string.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class ArtifactStore:
    """Directory-backed artifact store with a persistent sha256 manifest.

    Each ``save_*`` writes the artifact under ``root`` and records
    ``{filename: sha256}`` in ``root/manifest.json``. Reopening a store on
    an existing directory reloads the manifest.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / "manifest.json"
        self.manifest: dict[str, str] = {}
        if self._manifest_path.exists():
            self.manifest = json.loads(self._manifest_path.read_text())

    def path_for(self, filename: str) -> Path:
        """Return the absolute path for an artifact filename."""
        return self.root / filename

    def _record(self, filename: str) -> None:
        digest = sha256_hex(self.path_for(filename).read_bytes())
        self.manifest[filename] = digest
        self._manifest_path.write_text(json.dumps(self.manifest, sort_keys=True, indent=2))

    def _require(self, filename: str) -> Path:
        path = self.path_for(filename)
        if not path.exists():
            raise FileNotFoundError(f"artifact '{filename}' not found under {self.root}")
        expected = self.manifest.get(filename)
        if expected is not None:
            actual = sha256_hex(path.read_bytes())
            if actual != expected:
                raise ValueError(
                    f"artifact '{filename}' failed integrity check: manifest records "
                    f"sha256 {expected[:12]}..., file has {actual[:12]}..."
                )
        return path

    def record_file(self, filename: str) -> None:
        """Register an externally-written file under ``root`` in the manifest.

        Used for artifacts whose owning object writes its own format (e.g.
        ``ReferencePartition.to_artifact``); the manifest still gets the hash.

        Raises:
            FileNotFoundError: If the file does not exist under ``root``.
        """
        if not self.path_for(filename).exists():
            raise FileNotFoundError(f"cannot record missing artifact '{filename}'")
        self._record(filename)

    def save_json(self, name: str, obj: Any) -> Path:
        """Save ``obj`` as canonical JSON under ``<name>.json``."""
        filename = f"{name}.json"
        self.path_for(filename).write_text(canonical_json(obj))
        self._record(filename)
        return self.path_for(filename)

    def load_json(self, name: str) -> Any:
        """Load an artifact written by :meth:`save_json`.

        Raises:
            FileNotFoundError: If the artifact does not exist.
        """
        return json.loads(self._require(f"{name}.json").read_text())

    def save_npz(self, name: str, **arrays: np.ndarray) -> Path:
        """Save named arrays under ``<name>.npz``."""
        filename = f"{name}.npz"
        # dict[str, Any] sidesteps a numpy-stubs quirk: ``**arrays`` must
        # otherwise type-check against savez's ``allow_pickle: bool`` kwarg.
        named: dict[str, Any] = dict(arrays)
        np.savez(self.path_for(filename), **named)
        self._record(filename)
        return self.path_for(filename)

    def load_npz(self, name: str) -> dict[str, np.ndarray]:
        """Load an artifact written by :meth:`save_npz`.

        Raises:
            FileNotFoundError: If the artifact does not exist.
        """
        with np.load(self._require(f"{name}.npz"), allow_pickle=False) as data:
            return dict(data.items())

    def save_joblib(self, name: str, obj: Any) -> Path:
        """Save an arbitrary picklable object under ``<name>.joblib``."""
        filename = f"{name}.joblib"
        joblib.dump(obj, self.path_for(filename))
        self._record(filename)
        return self.path_for(filename)

    def load_joblib(self, name: str) -> Any:
        """Load an artifact written by :meth:`save_joblib`.

        Security: joblib deserialization executes pickle bytecode, so this
        must only read artifacts the store itself wrote (local, self-produced
        benchmark artifacts -- the only v1 use). ``_require`` verifies the
        file's sha256 against the manifest before deserializing, so a
        swapped/tampered artifact fails loudly instead of executing.

        Raises:
            FileNotFoundError: If the artifact does not exist.
            ValueError: If the file's hash no longer matches the manifest.
        """
        return joblib.load(self._require(f"{name}.joblib"))
