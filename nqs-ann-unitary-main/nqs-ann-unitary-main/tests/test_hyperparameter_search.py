from scripts.hyperparameter_grid import build_cli_args, rank_global_candidates


def test_build_cli_args_handles_boolean_flags() -> None:
    args = build_cli_args("j1j2", {"use_sr": True, "n_rot": 10, "n_subspace": 8})
    assert "--no-sr" not in args
    assert "--n-rot" in args
    assert "10" in args
    assert "--n-subspace" in args
    assert "8" in args

    args = build_cli_args("j1j2", {"use_sr": False, "n_rot": 12})
    assert "--no-sr" in args
    assert "12" in args


def test_rank_global_candidates_builds_system_and_global_winners() -> None:
    rows = [
        {"system": "ising", "block": "training", "config": {"learning_rate": 1e-3, "use_sr": True}, "score": 12.5},
        {"system": "ising", "block": "training", "config": {"learning_rate": 2e-3, "use_sr": False}, "score": 10.0},
        {"system": "j1j2", "block": "training", "config": {"learning_rate": 1e-3, "use_sr": True}, "score": 15.0},
        {"system": "j1j2", "block": "training", "config": {"learning_rate": 3e-3, "use_sr": False}, "score": 13.0},
    ]

    summary = rank_global_candidates(rows)
    assert summary["per_system_best"]["ising"]["config"]["learning_rate"] == 1e-3
    assert summary["per_system_best"]["j1j2"]["config"]["learning_rate"] == 1e-3
    assert summary["global_best"]["mean_score"] >= 12.0
