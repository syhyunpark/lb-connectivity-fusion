#!/usr/bin/env python3
# export_network_maps_to_surface.py

import argparse
from pathlib import Path
import numpy as np


def load_npz(p: Path) -> dict:
    return dict(np.load(p, allow_pickle=True))


def get_basis_matrix(basis: dict, key_candidates=("phi", "Phi", "U", "eigvecs", "evecs")):
    for k in key_candidates:
        if k in basis:
            return np.asarray(basis[k])
    raise KeyError("No basis matrix found in basis file")


def maybe_unweight_area(U: np.ndarray, basis: dict, unweight: bool) -> np.ndarray:
    if not unweight:
        return U

    for ak in ("A", "area", "vertex_area", "area_weights"):
        if ak in basis:
            A = np.asarray(basis[ak]).reshape(-1)
            if A.size != U.shape[0]:
                raise ValueError("Area-weight length mismatch")
            return U / np.sqrt(np.maximum(A, 1e-12))[:, None]

    raise KeyError("No area weights found in basis file")


def split_hemis(V: int, n_lh: int | None) -> tuple[slice, slice]:
    if n_lh is None:
        n_lh = V // 2
    n_rh = V - n_lh
    return slice(0, n_lh), slice(n_lh, n_lh + n_rh)


def write_gifti(func_lh: np.ndarray, func_rh: np.ndarray, out_lh: Path, out_rh: Path):
    try:
        import nibabel as nib
        from nibabel.gifti import GiftiImage, GiftiDataArray
    except Exception as e:
        print("[WARN] nibabel not available; skipping GIFTI export:", e)
        return

    def one(vec: np.ndarray) -> "GiftiImage":
        g = GiftiImage()
        g.add_gifti_data_array(GiftiDataArray(vec.astype(np.float32)))
        return g

    nib.save(one(func_lh), str(out_lh))
    nib.save(one(func_rh), str(out_rh))


def main():
    ap = argparse.ArgumentParser("Export LB-network maps to cortical surface (npy + optional func.gii).")
    ap.add_argument("--fit_npz", required=True, help="fit npz containing Phi_hat")
    ap.add_argument("--basis_npz", required=True, help="LB eigenmodes npz containing basis matrix (e.g., phi)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--K", type=int, default=None, help="How many LB modes to use (default: K from fit Phi_hat)")
    ap.add_argument("--which_r", default="all", help="Comma list like '1,2,5' or 'all'")
    ap.add_argument("--n_lh", type=int, default=None, help="LH vertex count if basis is concatenated (default: V//2)")
    ap.add_argument("--unweight_area", action="store_true", help="If basis has area weights A, output unweighted maps")
    ap.add_argument("--formats", default="npy,gii", help="comma list: npy,gii")
    args = ap.parse_args()

    fit_p = Path(args.fit_npz).expanduser().resolve()
    bas_p = Path(args.basis_npz).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    fit = load_npz(fit_p)
    basis = load_npz(bas_p)

    Phi = np.asarray(fit["Phi_hat"], dtype=float)
    K_fit, R = Phi.shape

    K = args.K if args.K is not None else K_fit
    if K > K_fit:
        raise ValueError("K exceeds fitted rank")
    Phi = Phi[:K, :]

    U = get_basis_matrix(basis)
    if U.ndim != 2:
        raise ValueError("Basis matrix must be 2D")
    V, K_basis = U.shape
    if K > K_basis:
        raise ValueError("K exceeds basis rank")
    U = U[:, :K]

    U = maybe_unweight_area(U, basis, args.unweight_area)

    if args.which_r.strip().lower() == "all":
        r_list = list(range(1, R + 1))
    else:
        r_list = [int(x) for x in args.which_r.split(",") if x.strip()]
        for r in r_list:
            if r < 1 or r > R:
                raise ValueError("Factor index out of range")

    fmt = {x.strip().lower() for x in args.formats.split(",") if x.strip()}

    lh_sl, rh_sl = split_hemis(V, args.n_lh)

    for r in r_list:
        w = Phi[:, r - 1]
        m = U @ w
        m = m.astype(np.float32)

        if "npy" in fmt:
            np.save(outdir / f"net_r{r:02d}_map.npy", m)

        if "gii" in fmt:
            m_lh = m[lh_sl]
            m_rh = m[rh_sl]
            write_gifti(
                m_lh,
                m_rh,
                outdir / f"net_r{r:02d}_lh.func.gii",
                outdir / f"net_r{r:02d}_rh.func.gii",
            )

        print("wrote r=", r)

    meta = {
        "fit_npz": str(fit_p),
        "basis_npz": str(bas_p),
        "K_used": int(K),
        "R": int(R),
        "V": int(V),
        "n_lh": int(lh_sl.stop - lh_sl.start),
        "unweight_area": bool(args.unweight_area),
        "formats": sorted(list(fmt)),
    }
    (outdir / "export_meta.json").write_text(__import__("json").dumps(meta, indent=2))
    print("Done. Outdir:", outdir)


if __name__ == "__main__":
    main()