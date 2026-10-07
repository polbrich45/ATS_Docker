from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from itertools import product
from pathlib import Path

SYSTEMS = {
    "j1j2": {
        "hamiltonian": "j1j2",
        "lattice": "square",
        "lx": 4,
        "ly": 4,
        "j1": 1.0,
        "j2": 0.5,
    },
    "ising": {
        "hamiltonian": "ising",
        "lattice": "square",
        "lx": 4,
        "ly": 4,
        "hx": 0.5,
        "hz": 0.5,
    },
    "kitaev_gamma": {
        "hamiltonian": "kitaev_gamma",
        "lattice": "honeycomb",
        "lx": 2,
        "ly": 2,
        "K": 1.0,
        "gamma": 0.0,
    },
}

SEARCH_SPACE = {
    "architecture": {
        "width": [32, 48, 64],
        "depth": [2, 3, 4],
        "heads": [4, 8],
        "phase_scale": [0.0, 0.5, 1.0],
    },
    "training": {
        "learning_rate": [1e-3, 3e-3, 1e-2, 2e-2],
        "n_samples": [128, 256, 512],
        "n_nqs": [1, 2, 4],
        "use_sr": [False, True],
    },
    "rotation": {
        "n_rot": [25, 50, 100],
        "n_subspace": [8, 16, 32],
        "rotation_learning_rate": [1e-2, 5e-2, 1e-1],
        "participation_weight": [0.0, 0.5, 1.0],
    },
    "teacher": {
        "n_teacher": [1, 2, 4],
        "teacher_learning_rate": [1e-3, 3e-3, 1e-2],
        "teacher_phase_weight": [0.0, 0.1, 0.5],
    },
    "adaptive": {
        "n_outer": [1, 2, 3],
        "n_recovery": [0, 1, 2],
        "acceptance_sigma": [0.05, 0.1, 0.2],
    },
}

REPO_ROOT = Path(__file__).resolve().parents[1]


def rank_global_candidates(results: list[dict]) -> dict:
    """Aggregate per-system winners and a global best candidate across systems."""
    per_system_best = {}
    for system_name in sorted({row["system"] for row in results}):
        system_rows = [row for row in results if row["system"] == system_name]
        if not system_rows:
            continue
        per_system_best[system_name] = max(system_rows, key=lambda row: row["score"])

    grouped: dict[tuple, list[dict]] = {}
    for row in results:
        signature = tuple(sorted(row["config"].items()))
        grouped.setdefault(signature, []).append(row)

    global_entries = []
    for signature, rows in grouped.items():
        score_values = [row["score"] for row in rows]
        config = dict(signature)
        global_entries.append({
            "config": config,
            "mean_score": sum(score_values) / len(score_values),
            "systems": sorted({row["system"] for row in rows}),
            "score_values": score_values,
        })

    global_best = max(global_entries, key=lambda item: item["mean_score"]) if global_entries else {}
    return {
        "per_system_best": per_system_best,
        "global_best": global_best,
    }


def build_grid_candidates(system_name: str, block: str, limit: int | None = None):
    values = SEARCH_SPACE[block]
    keys = list(values)
    candidates = []
    for combo in product(*(values[k] for k in keys)):
        cfg = {keys[i]: combo[i] for i in range(len(keys))}
        candidates.append({
            "system": system_name,
            "block": block,
            "config": cfg,
            "base": SYSTEMS[system_name],
        })
        if limit is not None and len(candidates) >= limit:
            break
    return candidates


def build_random_candidates(system_name: str, block: str, n_samples: int = 20, seed: int = 0):
    values = SEARCH_SPACE[block]
    rng = random.Random(seed)
    candidates = []
    for _ in range(n_samples):
        cfg = {k: rng.choice(v) for k, v in values.items()}
        candidates.append({
            "system": system_name,
            "block": block,
            "config": cfg,
            "base": SYSTEMS[system_name],
        })
    return candidates


def proxy_score(config):
    """Lightweight heuristic for ranking candidates before execution."""
    score = 0.0
    for key, value in config.items():
        if key in {"width", "depth", "heads", "n_nqs", "n_rot", "n_subspace", "n_outer", "n_teacher"}:
            score += float(value) * 0.05
        elif key in {"learning_rate", "rotation_learning_rate", "teacher_learning_rate"}:
            score += 10.0 / max(float(value), 1e-4)
        elif key in {"n_samples", "phase_scale"}:
            score += float(value) * 0.2
        elif key == "use_sr" and value is True:
            score += 1.5
        elif key.startswith("teacher") and "weight" in key:
            score += float(value) * 1.5
    return score


def build_cli_args(system_name: str, config: dict) -> list[str]:
    system = SYSTEMS[system_name]
    cli = [
        "--lattice", system["lattice"],
        "--hamiltonian", system["hamiltonian"],
        "--lx", str(system["lx"]),
        "--ly", str(system["ly"]),
        "--pbc",
    ]
    if "j1" in system:
        cli += ["--j1", str(system["j1"])]
    if "j2" in system:
        cli += ["--j2", str(system["j2"])]
    if "hx" in system:
        cli += ["--hx", str(system["hx"])]
    if "hz" in system:
        cli += ["--hz", str(system["hz"])]
    if "K" in system:
        cli += ["--K", str(system["K"])]
    if "gamma" in system:
        cli += ["--gamma", str(system["gamma"])]

    alias = {
        "width": "--width",
        "depth": "--depth",
        "heads": "--heads",
        "phase_scale": "--phase-scale",
        "learning_rate": "--learning-rate",
        "n_samples": "--n-samples",
        "n_nqs": "--n-nqs",
        "use_sr": "--no-sr",
        "n_rot": "--n-rot",
        "n_subspace": "--n-subspace",
        "rotation_learning_rate": "--rotation-learning-rate",
        "participation_weight": "--participation-weight",
        "n_teacher": "--n-teacher",
        "teacher_learning_rate": "--teacher-learning-rate",
        "teacher_phase_weight": "--teacher-phase-weight",
        "n_outer": "--n-outer",
        "n_recovery": "--n-recovery",
        "acceptance_sigma": "--acceptance-sigma",
    }

    for key, value in config.items():
        if key == "use_sr":
            if value is False:
                cli.append("--no-sr")
            continue
        if value is False:
            continue
        if value is True:
            cli.append(alias[key])
            continue
        cli.extend([alias[key], str(value)])
    return cli


def search_candidates(strategy: str, system_name: str, block: str, n_samples: int, seed: int = 0):
    if strategy == "grid":
        return build_grid_candidates(system_name, block, limit=n_samples)
    if strategy == "random":
        return build_random_candidates(system_name, block, n_samples=n_samples, seed=seed)
    raise ValueError(f"Unknown strategy: {strategy}")


def _candidate_id(system_name: str, block: str, config: dict) -> str:
    tokens = [system_name, block]
    for key in sorted(config):
        tokens.append(f"{key}={config[key]}")
    return "-".join(str(token) for token in tokens)


def run_single_candidate(system_name: str, block: str, config: dict, *, quick: bool = True, output_root: str | Path = "runs/hyperparameter_search", seed: int = 0):
    candidate_dir = Path(output_root) / _candidate_id(system_name, block, config)
    command = [
        sys.executable,
        str(REPO_ROOT / "run.py"),
        *build_cli_args(system_name, config),
        "--output-dir",
        str(candidate_dir),
        "--seed",
        str(seed),
        "--display",
        "silent",
    ]
    if quick:
        command.append("--quick")

    process = subprocess.run(command, cwd=str(REPO_ROOT), capture_output=True, text=True)
    summary_path = candidate_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}

    final_energy = summary.get("final_energy")
    return {
        "system": system_name,
        "block": block,
        "config": config,
        "score": proxy_score(config),
        "final_energy": final_energy,
        "returncode": process.returncode,
        "output_dir": str(candidate_dir),
        "stderr": process.stderr.strip(),
        "stdout": process.stdout.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple hyperparameter search for the adaptive NQS workflow.")
    parser.add_argument("--system", choices=sorted(SYSTEMS) + ["all"], default="j1j2")
    parser.add_argument("--block", choices=sorted(SEARCH_SPACE) + ["all"], default="architecture")
    parser.add_argument("--strategy", choices=("grid", "random"), default="grid")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="results/hyperparameter_candidates.json")
    parser.add_argument("--evaluate", action="store_true", help="Actually execute each candidate with run.py and save the resulting metrics.")
    parser.add_argument("--quick", action="store_true", help="Use the quick smoke-test configuration for each trial to keep the search cheap.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    system_names = sorted(SYSTEMS) if args.system == "all" else [args.system]
    blocks = sorted(SEARCH_SPACE) if args.block == "all" else [args.block]

    all_ranked = []
    for system_name in system_names:
        for block in blocks:
            candidates = search_candidates(args.strategy, system_name, block, args.samples, seed=args.seed)
            ranked_candidates = sorted(candidates, key=lambda item: proxy_score(item["config"]), reverse=True)
            all_ranked.extend([
                {"system": system_name, "block": block, "config": item["config"], "score": proxy_score(item["config"])}
                for item in ranked_candidates[: args.top_k]
            ])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.evaluate:
        summary = rank_global_candidates(all_ranked)
        output_path.write_text(json.dumps({
            "systems": system_names,
            "blocks": blocks,
            "per_system_best": {
                key: {"system": value["system"], "block": value["block"], "config": value["config"], "score": value["score"]}
                for key, value in summary["per_system_best"].items()
            },
            "global_best": summary["global_best"],
            "ranked": sorted(all_ranked, key=lambda row: row["score"], reverse=True)[: args.top_k],
        }, indent=2), encoding="utf-8")
        print(f"strategy={args.strategy} systems={system_names} blocks={blocks}")
        print(f"candidates={len(all_ranked)}")
        print(f"top candidates saved to: {output_path}")
        for item in sorted(all_ranked, key=lambda row: row["score"], reverse=True)[:5]:
            print(f"system={item['system']} block={item['block']} score={item['score']:.3f} config={item['config']}")

        if summary["global_best"]:
            print("\nBest global candidate across all systems:")
            print(json.dumps(summary["global_best"], indent=2, default=str))
        raise SystemExit(0)

    results = []
    for system_name in system_names:
        for block in blocks:
            candidates = search_candidates(args.strategy, system_name, block, args.samples, seed=args.seed)
            ranked_candidates = sorted(candidates, key=lambda item: proxy_score(item["config"]), reverse=True)
            for item in ranked_candidates[: args.top_k]:
                result = run_single_candidate(system_name, block, item["config"], quick=args.quick, output_root="runs/hyperparameter_search", seed=args.seed)
                results.append(result)

    ordered = sorted(results, key=lambda row: (row["final_energy"] is None, row["final_energy"] if row["final_energy"] is not None else float("inf")))
    output_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")

    print(f"strategy={args.strategy} systems={system_names} blocks={blocks}")
    print(f"evaluated={len(ordered)}")
    print(f"result file: {output_path}")
    for row in ordered[:5]:
        if row["final_energy"] is None:
            print(f"system={row['system']} block={row['block']} candidate={row['config']} status=failed returncode={row['returncode']}")
        else:
            print(f"system={row['system']} block={row['block']} final_energy={row['final_energy']:+.8f} config={row['config']}")


if __name__ == "__main__":
    main()
