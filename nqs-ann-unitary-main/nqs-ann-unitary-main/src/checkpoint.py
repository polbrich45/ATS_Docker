"""Compact parameter, history, and outer-boundary checkpoint I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import dataclasses as dc
import json
from collections.abc import Mapping

import h5py
import numpy as np


def save_params(group: h5py.Group, params: Any) -> None:
	''' Validate and save a nested dict/list/tuple of NumPy arrays to an HDF5 group.'''
	def write_leaf(name: str, value: Any) -> None:
		array 	= np.asarray(value)
		kwargs 	= {"compression": "gzip"} if array.ndim > 0 else {}
		group.create_dataset(name, data=array, **kwargs)

	if isinstance(params, Mapping):
		for key, val in params.items():
			save_params(group.require_group(str(key)), val) if isinstance(val, (Mapping, list, tuple)) else write_leaf(str(key), val)
	elif isinstance(params, (list, tuple)):
		for i, val in enumerate(params):
			save_params(group.require_group(str(i)), val) if isinstance(val, (Mapping, list, tuple)) else write_leaf(str(i), val)
	else:
		write_leaf("_leaf", params)


def load_params(group: h5py.Group) -> Any:
	if "_leaf" in group:
		return np.asarray(group["_leaf"])
	return {key: load_params(item) if isinstance(item, h5py.Group) else np.asarray(item) for key, item in group.items()}


def save_weights(path: str | Path, params: Any) -> Path:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with h5py.File(path, "w") as f:
		save_params(f.require_group("weights"), params)
	return path


def load_weights(path: str | Path) -> Any:
	with h5py.File(Path(path), "r") as f:
		return load_params(f["weights"])


def json_dumps(payload: Any) -> str:
	"""Serialize configs, enums, NumPy values, and paths as stable JSON."""
	def convert(value: Any) -> Any:
		if dc.is_dataclass(value):
			return {field.name: convert(getattr(value, field.name)) for field in dc.fields(value)}
		if hasattr(value, "value"):
			return convert(value.value)
		if isinstance(value, Path):
			return str(value)
		if isinstance(value, np.ndarray):
			return convert(value.tolist())
		if isinstance(value, np.generic):
			return convert(value.item())
		if isinstance(value, float) and not np.isfinite(value):
			return None
		if isinstance(value, dict):
			return {str(key): convert(item) for key, item in value.items()}
		if isinstance(value, (tuple, list)):
			return [convert(item) for item in value]
		return value
	return json.dumps(convert(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_json(path: str | Path, payload: Any) -> Path:
	"""Write one human-readable JSON artifact."""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json_dumps(payload))
	return path


def save_outer_checkpoint(root: str | Path | None, outer: int, params: Any, theta: Any, key: Any, basis: Any) -> None:
	"""Overwrite the single restart checkpoint at a completed outer boundary."""
	if root is None:
		return
	out = Path(root) / "checkpoint"
	out.mkdir(parents=True, exist_ok=True)
	save_weights(out / "weights.h5", params)
	np.savez(
		out / "state.npz",
		outer	= np.asarray(outer, dtype=np.int64),
		theta	= np.asarray(theta, dtype=np.float64),
		key		= np.asarray(key),
		basis	= np.asarray(basis, dtype=np.int8),
	)


def load_outer_checkpoint(path: str | Path) -> tuple[int, Any, np.ndarray, np.ndarray, np.ndarray]:
	"""Load the single restart checkpoint and return the next outer index."""
	path = Path(path)
	path = path if path.name == "checkpoint" else path / "checkpoint"
	state_path = path / "state.npz"
	if not state_path.exists() or not (path / "weights.h5").exists():
		raise FileNotFoundError(f"Incomplete restart checkpoint: {path}")
	with np.load(state_path) as state:
		outer 	= int(state["outer"])
		theta 	= np.asarray(state["theta"], dtype=np.float64)
		key 	= np.asarray(state["key"])
		basis 	= np.asarray(state["basis"], dtype=np.int8)
	return outer + 1, load_weights(path / "weights.h5"), theta, key, basis


def validate_restart_config(path: str | Path, expected: Mapping[str, Any]) -> None:
	"""Require matching physics and parameter conventions; schedules may change.

	``expected`` maps top-level ``config.json`` entries to their current values.
	The entry ``circuit`` is read from the saved ``adaptive`` block.
	"""
	restart		= Path(path)
	config_path = (restart.parent if restart.name == "checkpoint" else restart) / "config.json"
	if not config_path.exists():
		raise ValueError(f"Restart is missing the run config: {config_path}")
	saved				= json.loads(config_path.read_text())
	saved["circuit"]	= saved.get("adaptive", {}).get("circuit")
	for name, value in json.loads(json_dumps(dict(expected))).items():
		if saved.get(name) != value:
			raise ValueError(f"Restart {name} differs from the saved configuration; only training schedules may change")

# --------------------------
#! EOF
# --------------------------
