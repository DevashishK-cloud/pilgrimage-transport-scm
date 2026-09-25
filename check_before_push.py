"""
check_before_push.py -- refuse to let licensed data reach a public repo.

Run this BEFORE `git add`, and again before `git push`:

    python check_before_push.py

Git history is permanent. A licensed file committed once and deleted in a
later commit is still recoverable from the repository forever, and the only
real fix is rewriting history or deleting the repo. This script is cheap
insurance against a mistake that cannot be undone.

Exit code 0 = safe, 1 = do not push.
"""

import subprocess
import sys
from pathlib import Path

# Anything matching these is licensed CPHS microdata or derived from it.
FORBIDDEN_PATTERNS = [
    "household_income_",          # raw CPHS waves
    "cmie_income_raw",
    ".parquet",                   # the built panel
    "audit_by_month",             # per-wave CPHS sample counts
    "donor_screen",               # 844 rows of district sample sizes / SEs
    "level_breaks",               # 772 rows of income-derived statistics
    "_series.csv",                # district-month outcome series
    "_eventstudy",                # district-month gap series
    "_weights.csv",               # donor weights over CPHS districts
    "_sources.csv",
    "_composition.csv",
    "_donor_distance",
]

# Size guard: nothing legitimate in this repo is large. A big file is almost
# always data that should not be here.
MAX_MB = 5

# Results that ARE publishable: headline estimates and figures. Anything else
# under output/ is a bulk table derived from licensed microdata and stays out.
ALLOWED = {"output/results/summary.csv"}


def tracked_and_staged():
    """Every path git currently tracks or has staged."""
    out = set()
    for cmd in (["git", "ls-files"],
                ["git", "diff", "--cached", "--name-only"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, check=True)
            out.update(x for x in r.stdout.splitlines() if x.strip())
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return sorted(out)


def main():
    files = tracked_and_staged()
    if not files:
        print("No files tracked or staged yet.")
        print("Confirm .gitignore exists BEFORE your first `git add`.")
        return 0

    problems = []
    for f in files:
        if f in ALLOWED:
            continue
        low = f.lower()
        for pat in FORBIDDEN_PATTERNS:
            if pat.lower() in low:
                problems.append((f, f"matches forbidden pattern {pat!r}"))
                break
        else:
            p = Path(f)
            if p.exists():
                mb = p.stat().st_size / 1e6
                if mb > MAX_MB:
                    problems.append((f, f"{mb:.1f} MB exceeds the {MAX_MB} MB guard"))

    print(f"Checked {len(files)} tracked/staged file(s).\n")
    if problems:
        print("DO NOT PUSH. The following must be removed first:\n")
        for f, why in problems:
            print(f"  {f}\n      {why}")
        print("\nTo unstage a file that has not been committed yet:")
        print("    git rm --cached <file>")
        print("\nIf it is ALREADY COMMITTED, unstaging is not enough -- the file")
        print("remains in history. Use git-filter-repo, or delete the remote")
        print("repository and start again with .gitignore in place first.")
        return 1

    print("Clean. No licensed data or oversized files staged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
