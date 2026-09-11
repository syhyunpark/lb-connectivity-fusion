#!/usr/bin/env python3
"""
observation_model_grid.py

Run the modality-specific observation-model validation from a CSV manifest... Every fit uses
multiple deterministic starts, and the solution with the smallest final
objective is retained.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_int_or_range(s: str) -> List[int]:
    out: List[int] = []
    for token in [x.strip() for x in s.split(",") if x.strip()]:
        if "-" in token:
            a, b = token.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(token))
    return out


def safe_token(x: object) -> str:
    return str(x).replace(".", "p").replace("-", "m").replace(" ", "_")


def run_cmd(cmd: List[str], log_path: Path | None, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    if log_path is None:
        subprocess.run(cmd, check=True)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as log:
            log.write("CMD: " + " ".join(cmd) + "\n\n")
            log.flush()
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)


def required(row: pd.Series, name: str) -> Any:
    if name not in row or pd.isna(row[name]):
        raise ValueError(f"Manifest row is missing {name}")
    return row[name]


def variant_spec(variant: str) -> Dict[str, Any]:
    if variant == "fused":
        return {"fit_mode": "fused", "w_eeg": "-1", "w_fmri": "1", "use_clean": False}
    if variant == "clean_fused":
        return {"fit_mode": "fused", "w_eeg": "-1", "w_fmri": "1", "use_clean": True}
    if variant == "eeg_only":
        return {"fit_mode": "eeg_only", "w_eeg": "-1", "w_fmri": "0", "use_clean": False}
    if variant == "fmri_only":
        return {"fit_mode": "fmri_only", "w_eeg": "0", "w_fmri": "1", "use_clean": False}
    raise ValueError(f"Unknown fit variant: {variant}")


def start_specs(fit_mode: str, seed: int, n_starts: int) -> List[Tuple[str, int]]:
    if n_starts < 1:
        raise ValueError("n_starts must be at least 1")
    out: List[Tuple[str, int]] = []
    if fit_mode in {"fused", "fmri_only"}:
        out.append(("fmri_eig", seed))
        for j in range(1, n_starts):
            out.append(("random", seed + 100_000 * j))
    else:
        for j in range(n_starts):
            out.append(("random", seed + 100_000 * (j + 1)))
    return out


def save_best_multistart(
    start_paths: List[Path],
    start_meta: List[Tuple[str, int]],
    final_path: Path,
    selection_path: Path,
) -> None:
    objectives = []
    for path in start_paths:
        with np.load(path, allow_pickle=True) as z:
            objectives.append(float(np.asarray(z["J_final"])))
    obj = np.asarray(objectives, dtype=float)
    if not np.any(np.isfinite(obj)):
        raise RuntimeError(f"No finite multistart objective for {final_path.name}")
    best = int(np.nanargmin(obj))
    with np.load(start_paths[best], allow_pickle=True) as z:
        payload = {k: z[k] for k in z.files}
    payload.update(
        {
            "multistart_objectives": obj,
            "multistart_methods": np.asarray([x[0] for x in start_meta], dtype=object),
            "multistart_seeds": np.asarray([x[1] for x in start_meta], dtype=np.int64),
            "multistart_best_index": np.array(best, dtype=np.int32),
            "multistart_best_method": np.array(start_meta[best][0], dtype=object),
            "multistart_best_seed": np.array(start_meta[best][1], dtype=np.int64),
        }
    )
    final_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(final_path, **payload)

    selection = {
        "final_fit": final_path.name,
        "best_index": best,
        "best_method": start_meta[best][0],
        "best_seed": start_meta[best][1],
        "objectives": objectives,
        "start_files": [p.name for p in start_paths],
    }
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    with open(selection_path, "w") as f:
        json.dump(selection, f, indent=2)
    print(f"  selected start {best + 1}/{len(start_paths)}: J={obj[best]:.6e}")


def collect_evaluations(eval_dir: Path, out_csv: Path) -> None:
    rows = []
    for path in sorted(eval_dir.glob("*_eval.json")):
        try:
            with open(path) as f:
                rows.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Wrote {out_csv} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run observation-model validation simulation suite")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seeds", default="1001-1030")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--sim_script", default="observation_model_simulation.py")
    ap.add_argument("--fit_script", default="shared_spatial_model.py")
    ap.add_argument("--eval_script", default="observation_model_evaluation.py")

    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--R_true", type=int, default=5)
    ap.add_argument("--R_fit", type=int, default=5)
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--fs", type=float, default=200.0)
    ap.add_argument("--tr", type=float, default=1.4)
    ap.add_argument("--kmax_eeg", type=int, default=20)
    ap.add_argument("--taper_s", type=float, default=3.0)
    ap.add_argument("--eeg_window_sec", type=float, default=10.0)
    ap.add_argument("--simulation_n_jobs", type=int, default=4)

    ap.add_argument("--max_iter", type=int, default=50)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--alpha_lambda", type=float, default=0.0)
    ap.add_argument("--alpha0", type=float, default=1e-6)
    ap.add_argument("--step_phi", type=float, default=1e-2)
    ap.add_argument("--fit_n_jobs", type=int, default=4)
    ap.add_argument("--lam_solver", choices=["bvls", "lsq"], default="bvls")
    ap.add_argument("--n_starts", type=int, default=3)

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = pd.read_csv(manifest_path)
    required_cols = {
        "scenario_id",
        "generator_scenario",
        "summary_snr",
        "k0_phi",
        "mask",
        "eeg_candidate_modes",
        "topology_angle_deg",
        "fit_variants",
    }
    missing = sorted(required_cols.difference(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")

    outdir = Path(args.outdir).expanduser().resolve()
    sim_dir = outdir / "sim"
    fit_dir = outdir / "fit"
    starts_dir = outdir / "fit_starts"
    selection_dir = outdir / "multistart_selection"
    eval_dir = outdir / "eval"
    factor_dir = outdir / "per_factor"
    bench_dir = outdir / "benchmark"
    log_dir = outdir / "logs"
    for directory in [
        sim_dir,
        fit_dir,
        starts_dir,
        selection_dir,
        eval_dir,
        factor_dir,
        bench_dir,
        log_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_or_range(args.seeds)
    print(f"Simulation data sets: {len(manifest) * len(seeds)}")
    n_final_fits = 0
    for _, row in manifest.iterrows():
        n_final_fits += len([x for x in str(row["fit_variants"]).split(";") if x.strip()]) * len(seeds)
    print(f"Final fit/evaluation variants: {n_final_fits}")
    print(f"Underlying optimizer runs with multistart: {n_final_fits * args.n_starts}")

    total_sim = len(manifest) * len(seeds)
    sim_index = 0
    for _, row in manifest.iterrows():
        scenario_id = str(required(row, "scenario_id"))
        variants = [x.strip() for x in str(required(row, "fit_variants")).split(";") if x.strip()]
        for seed in seeds:
            sim_index += 1
            tag = f"{safe_token(scenario_id)}_seed{seed}"
            sim_path = sim_dir / f"{tag}.npz"
            benchmark_cache = bench_dir / f"{tag}_benchmarks.json"
            print(f"\n[{sim_index}/{total_sim}] {tag}", flush=True)

            need_sim = args.overwrite or not sim_path.exists()
            if not args.resume:
                need_sim = True
            if need_sim:
                sim_cmd = [
                    args.python,
                    args.sim_script,
                    "--out",
                    str(sim_path),
                    "--scenario_label",
                    scenario_id,
                    "--scenario",
                    str(required(row, "generator_scenario")),
                    "--n",
                    str(args.n),
                    "--K",
                    str(args.K),
                    "--R_true",
                    str(args.R_true),
                    "--duration",
                    str(args.duration),
                    "--fs",
                    str(args.fs),
                    "--tr",
                    str(args.tr),
                    "--summary_snr",
                    str(float(required(row, "summary_snr"))),
                    "--seed",
                    str(seed),
                    "--n_jobs",
                    str(args.simulation_n_jobs),
                    "--k0_phi",
                    str(float(required(row, "k0_phi"))),
                    "--kmax_eeg",
                    str(args.kmax_eeg),
                    "--mask",
                    str(required(row, "mask")),
                    "--taper_s",
                    str(args.taper_s),
                    "--eeg_candidate_modes",
                    str(int(required(row, "eeg_candidate_modes"))),
                    "--topology_angle_deg",
                    str(float(required(row, "topology_angle_deg"))),
                    "--eeg_window_sec",
                    str(args.eeg_window_sec),
                    "--modality_scaling",
                    "robust",
                ]
                run_cmd(sim_cmd, log_dir / f"{tag}_simulate.log" if args.log else None, args.dry_run)
            else:
                print("  simulation exists; skipping")

            final_paths: Dict[str, Path] = {}
            # Fit all variants first so the noisy fused evaluation can reference clean_fused.
            for variant in variants:
                spec = variant_spec(variant)
                final_path = fit_dir / f"{tag}_{variant}_fit.npz"
                selection_path = selection_dir / f"{tag}_{variant}_selection.json"
                final_paths[variant] = final_path
                need_final = args.overwrite or not final_path.exists()
                if not args.resume:
                    need_final = True
                if not need_final:
                    print(f"  {variant} final fit exists; skipping")
                    continue

                starts = start_specs(spec["fit_mode"], seed, args.n_starts)
                start_paths: List[Path] = []
                for start_idx, (method, start_seed) in enumerate(starts, start=1):
                    start_path = starts_dir / f"{tag}_{variant}_start{start_idx}_{method}_fit.npz"
                    start_paths.append(start_path)
                    need_start = args.overwrite or not start_path.exists()
                    if not args.resume:
                        need_start = True
                    if need_start:
                        fit_cmd = [
                            args.python,
                            args.fit_script,
                            "--data",
                            str(sim_path),
                            "--out",
                            str(start_path),
                            "--R",
                            str(args.R_fit),
                            "--fmri_mode",
                            "separate",
                            "--w_eeg",
                            spec["w_eeg"],
                            "--w_fmri",
                            spec["w_fmri"],
                            "--max_iter",
                            str(args.max_iter),
                            "--tol",
                            str(args.tol),
                            "--alpha_lambda",
                            str(args.alpha_lambda),
                            "--alpha0",
                            str(args.alpha0),
                            "--step_phi",
                            str(args.step_phi),
                            "--n_jobs",
                            str(args.fit_n_jobs),
                            "--lam_solver",
                            args.lam_solver,
                            "--sort_factors",
                            "median",
                            "--init_method",
                            method,
                            "--init_seed",
                            str(start_seed),
                        ]
                        if spec["use_clean"]:
                            fit_cmd.append("--use_clean_data")
                        run_cmd(
                            fit_cmd,
                            log_dir / f"{tag}_{variant}_start{start_idx}_fit.log" if args.log else None,
                            args.dry_run,
                        )
                    else:
                        print(f"  {variant} start {start_idx} exists; skipping")

                if not args.dry_run:
                    save_best_multistart(start_paths, starts, final_path, selection_path)

            for variant in variants:
                spec = variant_spec(variant)
                fit_path = final_paths[variant]
                eval_json = eval_dir / f"{tag}_{variant}_eval.json"
                eval_csv = factor_dir / f"{tag}_{variant}_per_factor.csv"
                need_eval = args.overwrite or not eval_json.exists()
                if not args.resume:
                    need_eval = True
                if need_eval:
                    eval_cmd = [
                        args.python,
                        args.eval_script,
                        "--data",
                        str(sim_path),
                        "--fit",
                        str(fit_path),
                        "--fit_mode",
                        spec["fit_mode"],
                        "--fit_variant",
                        variant,
                        "--out_json",
                        str(eval_json),
                        "--out_csv",
                        str(eval_csv),
                        "--benchmark_cache",
                        str(benchmark_cache),
                    ]
                    if variant == "fused" and "clean_fused" in final_paths:
                        eval_cmd.extend(["--clean_reference_fit", str(final_paths["clean_fused"])])
                    run_cmd(eval_cmd, log_dir / f"{tag}_{variant}_eval.log" if args.log else None, args.dry_run)
                else:
                    print(f"  {variant} evaluation exists; skipping")

            if not args.dry_run:
                collect_evaluations(eval_dir, outdir / "observation_model_summary_partial.csv")

    if not args.dry_run:
        collect_evaluations(eval_dir, outdir / "observation_model_summary.csv")


if __name__ == "__main__":
    main()
