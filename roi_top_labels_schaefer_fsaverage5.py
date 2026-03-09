#!/usr/bin/env python3
"""
roi_top_labels_schaefer_fsaverage5.py

Compute top Schaefer fsaverage5 parcel labels for each network map.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional
import subprocess
import ssl
import urllib.request
import re

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.freesurfer.io import read_annot


def _pick_hemi_file(maps_dir: Path, r: int, hemi: str, prefer_pos: bool) -> Path:
    r2 = f"{r:02d}"
    if prefer_pos:
        cand = sorted(maps_dir.glob(f"net_r{r2}_{hemi}_pos.func.gii"))
        if cand:
            return cand[0]
    cand = sorted(maps_dir.glob(f"net_r{r2}_{hemi}.func.gii"))
    if cand:
        return cand[0]
    cand = sorted(maps_dir.glob(f"net_r{r2}_{hemi}*.func.gii"))
    if cand:
        return cand[0]
    raise FileNotFoundError("Missing map file")


def _read_gifti_scalar(p: Path) -> np.ndarray:
    g = nib.load(str(p))
    if not hasattr(g, "darrays") or len(g.darrays) == 0:
        raise ValueError("No darrays found")
    return np.asarray(g.darrays[0].data).reshape(-1).astype(np.float64)


def _cbig_url(hemi: str, n_parcels: int, n_networks: int) -> str:
    return (
        "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
        "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/"
        "FreeSurfer5.3/fsaverage5/label/"
        f"{hemi}.Schaefer2018_{n_parcels}Parcels_{n_networks}Networks_order.annot"
    )


def _download_url_to_file_urllib(url: str, out_path: Path, insecure: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if insecure:
        ctx = ssl._create_unverified_context()
    else:
        try:
            import certifi  # type: ignore
            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ctx = ssl.create_default_context()

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx) as r, open(out_path, "wb") as f:
        f.write(r.read())


def _download_url_to_file_curl(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["/usr/bin/curl", "-L", "-o", str(out_path), url]
    subprocess.run(cmd, check=True)


def _download_if_missing(url: str, out_path: Path, backend: str, insecure_ok: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return

    last_err: Optional[Exception] = None

    def try_urllib(insecure: bool):
        nonlocal last_err
        try:
            _download_url_to_file_urllib(url, out_path, insecure=insecure)
            return True
        except Exception as e:
            last_err = e
            return False

    def try_curl():
        nonlocal last_err
        try:
            _download_url_to_file_curl(url, out_path)
            return True
        except Exception as e:
            last_err = e
            return False

    print("[download]", url)
    if backend == "urllib":
        if not try_urllib(insecure=False) and insecure_ok:
            print("[warn] urllib failed; retrying with insecure SSL")
            if not try_urllib(insecure=True):
                raise RuntimeError("Download failed")
        elif out_path.exists() and out_path.stat().st_size > 0:
            return
        else:
            raise RuntimeError("Download failed")

    elif backend == "curl":
        if not try_curl():
            raise RuntimeError("Download failed")

    else:
        if try_urllib(insecure=False):
            return
        if try_curl():
            return
        if insecure_ok:
            print("[warn] retrying with insecure SSL")
            if try_urllib(insecure=True):
                return
        raise RuntimeError("Download failed")


def load_schaefer_fsaverage5_annot(
    cache_dir: Path,
    n_parcels: int,
    n_networks: int,
    download_backend: str,
    insecure_download: bool,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    lh_path = cache_dir / f"lh.Schaefer2018_{n_parcels}Parcels_{n_networks}Networks_order.annot"
    rh_path = cache_dir / f"rh.Schaefer2018_{n_parcels}Parcels_{n_networks}Networks_order.annot"

    _download_if_missing(_cbig_url("lh", n_parcels, n_networks), lh_path, download_backend, insecure_download)
    _download_if_missing(_cbig_url("rh", n_parcels, n_networks), rh_path, download_backend, insecure_download)

    labL, _, namesL = read_annot(str(lh_path))
    labR, _, namesR = read_annot(str(rh_path))

    namesL = [n.decode("utf-8", errors="ignore") if isinstance(n, (bytes, bytearray)) else str(n) for n in namesL]
    namesR = [n.decode("utf-8", errors="ignore") if isinstance(n, (bytes, bytearray)) else str(n) for n in namesR]

    return (labL, namesL), (labR, namesR)


_EXCLUDE_PAT = re.compile(r"(unknown|medialwall|medial_wall|\?\?\?|background|medial\s*wall)", re.IGNORECASE)


def _exclude_parcel(name: str) -> bool:
    if not isinstance(name, str):
        return True
    return bool(_EXCLUDE_PAT.search(name))


def score_parcels(values: np.ndarray, labels: np.ndarray, names: List[str], hemi: str) -> pd.DataFrame:
    if values.shape[0] != labels.shape[0]:
        raise ValueError("Vertex count mismatch")

    rows = []
    for lab in np.unique(labels):
        if lab < 0:
            continue
        nm = names[lab] if (0 <= lab < len(names)) else f"label_{lab}"
        if _exclude_parcel(nm):
            continue

        idx = labels == lab
        if idx.sum() == 0:
            continue
        v = values[idx]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue

        rows.append({
            "hemi": hemi,
            "parcel_id": int(lab),
            "parcel_name": nm,
            "n_vertices": int(idx.sum()),
            "mean": float(np.mean(v)),
            "abs_mean": float(np.mean(np.abs(v))),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["abs_mean"], ascending=False).reset_index(drop=True)
    return df


def pool_bilateral(dL: pd.DataFrame, dR: pd.DataFrame) -> pd.DataFrame:
    def strip_hemi(name: str) -> str:
        if not isinstance(name, str):
            return str(name)
        name = name.replace("7Networks_LH_", "").replace("7Networks_RH_", "")
        return name

    dd = pd.concat([dL.copy(), dR.copy()], axis=0, ignore_index=True)
    if dd.empty:
        return dd

    dd["parcel_name_stripped"] = dd["parcel_name"].astype(str).map(strip_hemi)

    def wmean(x, w):
        x = np.asarray(x, float)
        w = np.asarray(w, float)
        s = np.sum(w)
        return float(np.sum(w * x) / s) if s > 0 else float(np.nan)

    out_rows = []
    for pname, g in dd.groupby("parcel_name_stripped", sort=False):
        w = g["n_vertices"].to_numpy(float)
        out_rows.append({
            "hemi": "BI",
            "parcel_id": -1,
            "parcel_name": pname,
            "n_vertices": int(np.sum(w)),
            "mean": wmean(g["mean"].to_numpy(float), w),
            "abs_mean": wmean(g["abs_mean"].to_numpy(float), w),
            "n_hemi_parts": int(g.shape[0]),
        })

    out = pd.DataFrame(out_rows).sort_values("abs_mean", ascending=False).reset_index(drop=True)
    return out


def parse_networks(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser("Top ROI labels for each network map")
    ap.add_argument("--maps_dir", required=True)
    ap.add_argument("--networks", required=True, help="Comma list like 1,2,3")
    ap.add_argument("--prefer_pos", action="store_true")

    ap.add_argument("--n_parcels", type=int, default=200)
    ap.add_argument("--n_networks", type=int, default=7)

    ap.add_argument("--atlas_cache_dir", default="", help="Default: <maps_dir>/atlas_cache")
    ap.add_argument("--top_n", type=int, default=10)

    ap.add_argument(
        "--summary",
        choices=["bilat", "hemi", "both"],
        default="bilat",
        help="bilat=LH+RH pooled; hemi=LH/RH; both=all",
    )
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_txt", default="")

    ap.add_argument("--download_backend", choices=["auto", "curl", "urllib"], default="auto")
    ap.add_argument(
        "--insecure_download",
        action="store_true",
        help="Fallback only if certificate verification fails",
    )
    args = ap.parse_args()

    maps_dir = Path(args.maps_dir).expanduser().resolve()
    nets = parse_networks(args.networks)

    cache_dir = (
        Path(args.atlas_cache_dir).expanduser().resolve()
        if args.atlas_cache_dir.strip()
        else (maps_dir / "atlas_cache")
    )
    (labL, namesL), (labR, namesR) = load_schaefer_fsaverage5_annot(
        cache_dir,
        args.n_parcels,
        args.n_networks,
        download_backend=args.download_backend,
        insecure_download=bool(args.insecure_download),
    )

    all_rows = []
    txt_lines = []

    want_bilat = args.summary in ["bilat", "both"]
    want_hemi = args.summary in ["hemi", "both"]

    for r in nets:
        fL = _pick_hemi_file(maps_dir, r, "lh", args.prefer_pos)
        fR = _pick_hemi_file(maps_dir, r, "rh", args.prefer_pos)

        vL = _read_gifti_scalar(fL)
        vR = _read_gifti_scalar(fR)

        dL = score_parcels(vL, labL, namesL, "LH")
        dR = score_parcels(vR, labR, namesR, "RH")
        dB = pool_bilateral(dL, dR)

        if want_hemi:
            dL2 = dL.copy()
            dL2["network"] = r
            dR2 = dR.copy()
            dR2["network"] = r
            all_rows.append(dL2)
            all_rows.append(dR2)

        if want_bilat:
            dB2 = dB.copy()
            dB2["network"] = r
            all_rows.append(dB2)

        if args.out_txt:
            txt_lines.append(f"=== Network {r} (top {args.top_n} by abs_mean) ===")

            if want_bilat:
                txt_lines.append("  BI:")
                for j in range(min(args.top_n, len(dB))):
                    row = dB.iloc[j]
                    txt_lines.append(
                        f"    {j+1:2d}. {row['parcel_name']}  "
                        f"(abs_mean={row['abs_mean']:.4g}, mean={row['mean']:.4g}, n={int(row['n_vertices'])})"
                    )

            if want_hemi:
                for hemi, dfh in [("LH", dL), ("RH", dR)]:
                    txt_lines.append(f"  {hemi}:")
                    for j in range(min(args.top_n, len(dfh))):
                        row = dfh.iloc[j]
                        txt_lines.append(
                            f"    {j+1:2d}. {row['parcel_name']}  "
                            f"(abs_mean={row['abs_mean']:.4g}, mean={row['mean']:.4g}, n={int(row['n_vertices'])})"
                        )
            txt_lines.append("")

    if not all_rows:
        raise RuntimeError("No output rows produced")

    out = pd.concat(all_rows, axis=0, ignore_index=True)

    out_path = Path(args.out_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("Wrote:", out_path)

    if args.out_txt:
        txt_path = Path(args.out_txt).expanduser().resolve()
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text("\n".join(txt_lines))
        print("Wrote:", txt_path)


if __name__ == "__main__":
    main()