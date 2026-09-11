#!/usr/bin/env python3
"""
rank_screening_grid.py

 local runner:
    simulate -> ARD fit -> evaluate -> save one JSON per case 
"""


from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_seeds(s: str) -> List[int]:
    s = s.strip()
    if "-" in s and "," not in s:
        a, b = [int(x) for x in s.split("-", 1)]
        return list(range(a, b + 1))
    return parse_int_list(s)


def tag_float(x: float) -> str:
    return f"{x:g}".replace(".", "p").replace("-", "m")


def run_cmd(cmd: List[str], env: dict, log_path: Path, quiet: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if quiet:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        log_path.write_text(
            "$ " + " ".join(cmd) + "\n\nSTDOUT\n" + p.stdout
            + "\n\nSTDERR\n" + p.stderr
        )
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, cmd)
    else:
        with open(log_path, "w") as log:
            log.write("$ " + " ".join(cmd) + "\n\n")
            subprocess.run(
                cmd,
                check=True,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )




def run_case(case: Tuple, cfg: dict) -> str:
    trueR, n, noise_model, seed = case
    outdir = Path(cfg["outdir"])
    case_tag = (
        f"trueR{trueR}_n{n}_noise{noise_model}_"
        f"snr{tag_float(cfg['snr'])}_seed{seed}_Rmax{cfg['Rmax']}"
    )
    eval_path = outdir / "eval" / f"{case_tag}_eval.json"
    log_base = outdir / "logs" / case_tag

    if cfg["resume"] and eval_path.exists():
        try:
            json.loads(eval_path.read_text())
            return f"resume-skip: {case_tag}"
        except Exception:
            pass

    env = os.environ.copy()
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        env[key] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)

    if cfg["keep_artifacts"]:
        work = outdir / "artifacts" / case_tag
        work.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work = Path(tempfile.mkdtemp(prefix=case_tag + "_"))
        cleanup = True

    sim_path = work / "sim.npz"
    fit_path = work / "fit.npz"

    try:
        sim_cmd = [
            cfg["python"], cfg["sim_script"],
            "--R", str(trueR),
            "--kmax_eeg", str(cfg["kmax"]),
            "--snr", str(cfg["snr"]),
            "--seed", str(seed),
            "--n", str(n),
            "--K", str(cfg["K"]),
            "--F", str(cfg["F"]),
            "--noise_model", noise_model,
            "--t_df", str(cfg["t_df"]),
            "--hetero_gamma", str(cfg["hetero_gamma"]),
            "--out", str(sim_path),
        ]
        run_cmd(sim_cmd, env, log_base.with_suffix(".sim.log"), cfg["quiet"])

        fit_cmd = [
            cfg["python"], cfg["fit_script"],
            "--data", str(sim_path),
            "--out", str(fit_path),
            "--use_ard",
            "--Rmax", str(cfg["Rmax"]),
            "--fmri_mode", cfg["fmri_mode"],
            "--alpha_lambda", str(cfg["alpha_lambda"]),
            "--alpha0", str(cfg["alpha0"]),
            "--ard_eta", str(cfg["ard_eta"]),
            "--ard_burnin", str(cfg["ard_burnin"]),
            "--ard_a", str(cfg["ard_a"]),
            "--ard_b", str(cfg["ard_b"]),
            "--ard_tau_floor", str(cfg["ard_tau_floor"]),
            "--ard_tau_ceiling", str(cfg["ard_tau_ceiling"]),
            "--ard_energy_rel_thresh", str(cfg["ard_energy_rel_thresh"]),
            "--step_phi", str(cfg["step_phi"]),
            "--max_iter", str(cfg["max_iter"]),
            "--tol", str(cfg["tol"]),
            "--n_jobs", str(cfg["fit_n_jobs"]),
            "--lam_solver", cfg["lam_solver"],
            "--sort_factors", "median",
        ]
        run_cmd(fit_cmd, env, log_base.with_suffix(".fit.log"), cfg["quiet"])

        eval_cmd = [
            cfg["python"], cfg["eval_script"],
            "--sim", str(sim_path),
            "--fit", str(fit_path),
            "--json_out", str(eval_path),
            "--ard_rel_thresh", str(cfg["ard_energy_rel_thresh"]),
        ]
        run_cmd(eval_cmd, env, log_base.with_suffix(".eval.log"), cfg["quiet"])
    finally:
        if cleanup:
            shutil.rmtree(work, ignore_errors=True)

    return f"done: {case_tag}"


def main() -> None:
    ap = argparse.ArgumentParser("Run the adaptive ARD-motivated rank-screening simulation grid.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--trueR_list", default="5,10")
    ap.add_argument("--n_list", default="100,200")
    ap.add_argument(
        "--noise_models", default="gaussian,student_t,freq_hetero"
    )
    ap.add_argument("--seeds", default="1001-1020")
    ap.add_argument("--snr", type=float, default=2.0)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--F", type=int, default=20)
    ap.add_argument("--kmax", type=int, default=20)
    ap.add_argument("--Rmax", type=int, default=15)
    ap.add_argument(
        "--fmri_mode", choices=["separate", "aggregate"], default="separate"
    )
    ap.add_argument("--t_df", type=float, default=5.0)
    ap.add_argument("--hetero_gamma", type=float, default=0.5)
    ap.add_argument("--alpha_lambda", type=float, default=1.0)
    ap.add_argument("--alpha0", type=float, default=1e-6)
    ap.add_argument("--ard_eta", type=float, default=0.1)
    ap.add_argument("--ard_burnin", type=int, default=5)
    ap.add_argument("--ard_a", type=float, default=1e-6)
    ap.add_argument("--ard_b", type=float, default=1e-6)
    ap.add_argument("--ard_tau_floor", type=float, default=1e-6)
    ap.add_argument("--ard_tau_ceiling", type=float, default=1e8)
    ap.add_argument("--ard_energy_rel_thresh", type=float, default=1e-2)
    ap.add_argument("--step_phi", type=float, default=1e-2)
    ap.add_argument("--max_iter", type=int, default=30)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--fit_n_jobs", type=int, default=4)
    ap.add_argument("--case_jobs", type=int, default=1)
    ap.add_argument("--lam_solver", choices=["bvls", "lsq"], default="bvls")
    ap.add_argument("--sim_script", default="rank_screening_simulation.py")
    ap.add_argument("--fit_script", default="shared_spatial_model.py")
    ap.add_argument("--eval_script", default="rank_screening_evaluation.py")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--keep_artifacts", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    (outdir / "eval").mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(parents=True, exist_ok=True)

    for name in ["sim_script", "fit_script", "eval_script"]:
        p = Path(getattr(args, name)).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        setattr(args, name, str(p))

    trueRs = parse_int_list(args.trueR_list)
    ns = parse_int_list(args.n_list)
    noises = parse_str_list(args.noise_models)
    seeds = parse_seeds(args.seeds)
    cases = [
        (r, n, noise, seed)
        for r in trueRs
        for n in ns
        for noise in noises
        for seed in seeds
    ]

    cfg = vars(args).copy()
    cfg["outdir"] = str(outdir)
    cfg["python"] = sys.executable
    (outdir / "run_config.json").write_text(
        json.dumps(cfg, indent=2, default=str)
    )

    print(f"N cases: {len(cases)}")
    print(f"Output: {outdir}")
    print(
        "Avoid oversubscription: case_jobs * fit_n_jobs = "
        f"{args.case_jobs * args.fit_n_jobs}"
    )

    if args.case_jobs <= 1:
        for j, case in enumerate(cases, 1):
            print(f"[{j}/{len(cases)}] {case}")
            print(run_case(case, cfg))
    else:
        with ProcessPoolExecutor(max_workers=args.case_jobs) as ex:
            futures = {ex.submit(run_case, c, cfg): c for c in cases}
            done = 0
            for fut in as_completed(futures):
                done += 1
                case = futures[fut]
                try:
                    print(f"[{done}/{len(cases)}] {fut.result()}")
                except Exception as exc:
                    print(f"[{done}/{len(cases)}] FAILED {case}: {exc}")
                    raise

    print("Finished.")


if __name__ == "__main__":
    main()
