#!/usr/bin/env python
"""H2 — one-command reproduction of the canonical Diff-GVAE pipeline.

Runs, in order, the steps that produce the canonical r32 results:

  graph        build `data_ln_pc_ihc_g_r32.pt`
  validate     run review_fixes_2026_07/data_validation.py on the graph
  oof          pooled-OOF GVAE evaluation, seeds 42-46 (radiology-artifact-ablation/run_oof.py)
  gvae-latent  best-params ranked GVAE on r32 = latent source for DDPM
  ddpm         conditional latent DDPM chain (A2, A5, A3, A4)

Steps always execute in the canonical order above, regardless of the order given
to `--steps`.

The script is a thin, explicit driver over the existing research scripts so the
paper numbers are reproducible from a clean checkout with one command:

    .venv/bin/python reproduce_canonical.py                 # everything
    .venv/bin/python reproduce_canonical.py --dry-run       # print commands only
    .venv/bin/python reproduce_canonical.py --steps graph,oof

Notes:
- This is the *protocol* reproduction; the GVAE/DDPM pipeline is not fully
  deterministic across thread counts (see `canonical_r32_report.md`), so exact
  floats may differ slightly from the archived package. Run ids are fresh.
- `ddpm` consumes the run id written by `gvae-latent`
  (`research/2026-10-03-canonical-r32/output/bestparam_ranked_r32_run_id.txt`),
  so `gvae-latent` must run before `ddpm`.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CANON = "research/2026-10-03-canonical-r32"
ABL = "research/2026-10-03-radiology-artifact-ablation"
DATA = "data_ln_pc_ihc_g_r32.pt"

STEPS = ["graph", "validate", "oof", "gvae-latent", "ddpm"]
DEFAULT_SEEDS = "42,43,44,45,46"


def commands(step: str, python: str, seeds: list[int]) -> list[list[str]]:
    """Return the exact argv lists for one step (pure; unit-testable)."""
    if step == "graph":
        return [[python, "data/build_ln_pc_ihc_g.py", "--out", DATA,
                 "--drop-radiology-artifacts", "both"]]
    if step == "validate":
        return [[python, "review_fixes_2026_07/data_validation.py", DATA]]
    if step == "oof":
        return [[python, f"{ABL}/scripts/run_oof.py", "--data", DATA,
                 "--tag", "drop_both32", "--seed", str(s)] for s in seeds]
    if step == "gvae-latent":
        return [[python, f"{CANON}/scripts/run_bestparam_ranked_r32.py"]]
    if step == "ddpm":
        return [["bash", f"{CANON}/scripts/run_ddpm_r32.sh"]]
    raise ValueError(f"unknown step: {step}")


def main() -> int:
    ap = argparse.ArgumentParser(description="One-command canonical reproduction (H2).")
    ap.add_argument("--steps", default=",".join(STEPS),
                    help=f"comma list from {STEPS} (default: all)")
    ap.add_argument("--seeds", default=DEFAULT_SEEDS, help="OOF seeds for the 'oof' step")
    ap.add_argument("--python", default=os.environ.get("REPRO_PYTHON", ".venv/bin/python"),
                    help="python interpreter for the steps (default: .venv/bin/python)")
    ap.add_argument("--dry-run", action="store_true", help="print commands, do not run")
    args = ap.parse_args()

    requested = {s.strip() for s in args.steps.split(",") if s.strip()}
    unknown = sorted(requested - set(STEPS))
    if unknown:
        raise SystemExit(f"unknown step(s) {unknown}; choose from {STEPS}")
    # Always run in canonical prerequisite order, never the order the user typed.
    steps = [s for s in STEPS if s in requested]
    seeds = [int(s) for s in args.seeds.split(",")]
    if "ddpm" in steps and "gvae-latent" not in steps:
        raise SystemExit("'ddpm' needs the run id from 'gvae-latent'; include it.")

    if not args.dry_run and not (ROOT / DATA).exists() and "graph" not in steps:
        raise SystemExit(f"{DATA} not found; run the 'graph' step first.")

    env = dict(os.environ, OMP_NUM_THREADS="3", MKL_NUM_THREADS="3")
    for step in steps:
        print(f"\n=== step: {step} ===", flush=True)
        for cmd in commands(step, args.python, seeds):
            print("$ " + " ".join(cmd), flush=True)
            if args.dry_run:
                continue
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    print("\nREPRODUCE_DONE" + (" (dry-run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
