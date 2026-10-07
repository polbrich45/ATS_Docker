"""Logging must preserve nested display policies and truthful stage status."""

import pytest

from general_log import Logger
from src import logging_utils as logs


@pytest.fixture(autouse=True)
def isolated_logger(monkeypatch, capsys):
	monkeypatch.setenv("PYLOGCOLORS", "0")
	monkeypatch.setenv("PYLOGLEVEL", "INFO")
	monkeypatch.delenv("PYLOGFILE", raising=False)
	monkeypatch.setattr(logs, "_LOGGER", None)
	yield
	if logs._LOGGER is not None:
		# Pytest closes the call-phase capture stream before fixture teardown.
		for handler in tuple(logs._LOGGER.logger.handlers):
			logs._LOGGER.logger.removeHandler(handler)
			handler.close()


def test_general_log_uses_literal_tabs_and_restores_display(capsys, monkeypatch):
	monkeypatch.setenv("NQS_PROGRESS", "1")
	with logs.display_context("jupyter"):
		logs.log("visible", level=3)
		logs.log("too deep", level=4)
		with logs.quiet():
			with logs.display_context("plain"):
				logs.log("hidden")
				assert logs.progress_disabled()
		assert not logs.progress_disabled()
	logs.log("restored", level=4)
	assert isinstance(logs._LOGGER, Logger)
	output = capsys.readouterr().out
	assert "[INFO] \t\t\t->visible" in output
	assert "[INFO] \t\t\t\t->restored" in output
	assert "too deep" not in output and "hidden" not in output


def test_failed_section_rethrows_and_does_not_report_success(capsys):
	with pytest.raises(ValueError, match="bad state"):
		with logs.section("Train", level=1):
			raise ValueError("bad state")
	output = capsys.readouterr().out
	assert "[ERROR] \t->Train | failed after" in output
	assert "finished" not in output


def test_timing_summary_accounts_for_overhead(capsys):
	logs.timing_summary({"Train": .8, "Evaluate": .1}, 1., label="Timings")
	output = capsys.readouterr().out
	assert "Other / logging" in output and "0.100" in output
	assert "Total" in output and "1.000" in output
	assert "[WARNING]" not in output
	with logs.display_context("silent"):
		logs.timing_summary({"Train": .8}, 1., label="Hidden timings")
	assert capsys.readouterr().out == ""


def test_interrupt_warning_respects_silent_mode(capsys):
	def interrupted():
		yield 1
		raise KeyboardInterrupt
	assert list(logs.interruptible(interrupted())) == [1]
	assert "[WARNING] \t->interrupted -- returning partial result" in capsys.readouterr().out
	with logs.display_context("silent"):
		assert list(logs.interruptible(interrupted())) == [1]
	assert capsys.readouterr().out == ""


def test_adaptive_logs_local_evaluation_times_and_full_workflow(monkeypatch):
	"""Two outer steps and a final Ritz solve must all contribute to wall time."""
	import re
	import numpy as np
	from src import adaptive
	from src.config import AdaptiveConfig, AdaptiveStep, CircuitConfig, FinalConfig, LatticeConfig, ModelConfig, RotationConfig, TeacherConfig, TrainConfig

	measurements, messages = [], []
	monkeypatch.setattr(adaptive, "_log_nqs_metrics", lambda label, metrics, n_sites, elapsed, **kwargs: measurements.append(elapsed))
	monkeypatch.setattr(adaptive, "log", lambda message, **kwargs: messages.append(message))
	lattice = LatticeConfig(hamiltonian="ising", Lx=2, Ly=1, pbc=False, hx=.7)
	model = ModelConfig(width=4, depth=1, heads=1, mlp_ratio=1, conserve_total_sz=False)
	step = AdaptiveStep(train=TrainConfig(n_nqs=0, n_samples=16), rotation=RotationConfig(n_rot=0, n_subspace=2, n_seed_samples=32), teacher=TeacherConfig(n_teacher=0), n_recovery=0)
	config = AdaptiveConfig(circuit=CircuitConfig(n_layers=0), steps=(step, step), final=FinalConfig(solver="exact", ritz_size=4), display="silent")
	result = adaptive.run_adaptive_loop(lattice, model, config)
	# n_rot=0 leaves the angles unchanged, so each step reuses its before-rotation measurement.
	assert len(measurements) == 2
	assert np.isclose(sum(measurements), sum(row["timing"]["evaluation_seconds"] for row in result["outer_history"]))
	for row in result["outer_history"]:
		assert sum(row["timing"].values()) <= row["seconds"]
	total_message = next(message for message in messages if message.startswith("Total wall time:"))
	seconds = float(re.search(r"Total wall time: ([0-9.]+) s", total_message)[1])
	assert seconds + .001 >= sum(row["seconds"] for row in result["outer_history"])
	assert any(message.startswith("Expanded final Ritz: 4 states") for message in messages)
	assert result["final_reference"]["reused"]
