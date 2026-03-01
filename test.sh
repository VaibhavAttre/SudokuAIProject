#!/usr/bin/env bash
set -euo pipefail

# Draft AI tester for the Sudoku Python shell.
#
# What it checks:
#   1) Builds the shell with `make`.
#   2) Generates GUARANTEED-SOLVABLE Easy / Intermediate / Hard boards
#      that match the project's Draft AI dimensions and givens.
#   3) Generates a few intentionally UNSOLVABLE smoke-test boards.
#   4) Runs MRV + LCV + FC on every board with a per-board timeout.
#   5) Verifies any returned solution is a valid Sudoku solution and that
#      all original givens are preserved.
#   6) Reports completion-rate thresholds for Draft AI.
#
# Important limitation:
#   The official average-score requirement depends on the hidden teacher
#   backtrack baseline, so this script CANNOT reproduce the exact course
#   score. It does verify the observable parts you can check locally:
#   solve rate, crashes, timeouts, unsolvable handling, and solution validity.
#
# Usage:
#   bash test_draft_ai.sh /path/to/Sudoku_Python_Shell [boards_per_difficulty]
#
# Example:
#   bash test_draft_ai.sh ~/SudokuAIProject-main/Sudoku_Python_Shell 20

PROJECT_DIR="${1:-$(pwd)}"
BOARDS_PER_DIFFICULTY="${2:-20}"

TIMEOUT_EASY="${TIMEOUT_EASY:-15}"
TIMEOUT_INTERMEDIATE="${TIMEOUT_INTERMEDIATE:-25}"
TIMEOUT_HARD="${TIMEOUT_HARD:-40}"
TIMEOUT_UNSOLVABLE="${TIMEOUT_UNSOLVABLE:-10}"

if [[ -d "$PROJECT_DIR/src" ]]; then
  SHELL_ROOT="$PROJECT_DIR"
elif [[ -f "$PROJECT_DIR/Makefile" && -d "$PROJECT_DIR/Sudoku_Python_Shell/src" ]]; then
  SHELL_ROOT="$PROJECT_DIR/Sudoku_Python_Shell"
else
  echo "[ERROR] Could not find Sudoku_Python_Shell."
  echo "        Pass the shell root directory (the folder that contains Makefile, src, and bin)."
  exit 1
fi

SRC_DIR="$SHELL_ROOT/src"
BIN_DIR="$SHELL_ROOT/bin"
MAIN_PYC="$BIN_DIR/Main.pyc"

if [[ ! -f "$SHELL_ROOT/Makefile" ]]; then
  echo "[ERROR] Makefile not found in: $SHELL_ROOT"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 is required."
  exit 1
fi

if ! command -v timeout >/dev/null 2>&1; then
  echo "[ERROR] timeout command is required."
  echo "        On macOS, install coreutils or replace 'timeout' with 'gtimeout'."
  exit 1
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/draft-ai-test.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

BOARDS_DIR="$WORK_DIR/boards"
RESULTS_DIR="$WORK_DIR/results"
mkdir -p "$BOARDS_DIR/easy" "$BOARDS_DIR/intermediate" "$BOARDS_DIR/hard" "$BOARDS_DIR/unsolvable" "$RESULTS_DIR"

echo "[INFO] Shell root: $SHELL_ROOT"
echo "[INFO] Temp work dir: $WORK_DIR"
echo "[INFO] Boards per difficulty: $BOARDS_PER_DIFFICULTY"
echo

echo "[STEP] Building solver with make..."
(
  cd "$SHELL_ROOT"
  make >/dev/null
)

if [[ ! -f "$MAIN_PYC" ]]; then
  echo "[ERROR] Build finished but $MAIN_PYC was not created."
  exit 1
fi

echo "[STEP] Generating guaranteed-solvable boards + unsolvable smoke tests..."
WORK_DIR="$WORK_DIR" BOARDS_PER_DIFFICULTY="$BOARDS_PER_DIFFICULTY" python3 <<'PY'
import os
import random
from pathlib import Path

work_dir = Path(os.environ["WORK_DIR"])
boards_per = int(os.environ["BOARDS_PER_DIFFICULTY"])
boards_dir = work_dir / "boards"

# Draft AI board settings from the project spec.
# Easy:         P=Q=3, N=9,  givens=7
# Intermediate: P=3, Q=4, N=12, givens=11
# Hard:         P=Q=4, N=16, givens=20
settings = [
    ("easy", 3, 3, 7),
    ("intermediate", 3, 4, 11),
    ("hard", 4, 4, 20),
]

def pattern(p, q, r, c):
    n = p * q
    return (r * q + (r // p) + c) % n

def shuffled(seq, rng):
    seq = list(seq)
    rng.shuffle(seq)
    return seq

def make_solution(p, q, seed):
    rng = random.Random(seed)
    n = p * q

    row_groups = shuffled(range(q), rng)
    rows = [g * p + r for g in row_groups for r in shuffled(range(p), rng)]

    col_groups = shuffled(range(p), rng)
    cols = [g * q + c for g in col_groups for c in shuffled(range(q), rng)]

    nums = shuffled(range(1, n + 1), rng)

    board = [[nums[pattern(p, q, r, c)] for c in cols] for r in rows]
    return board

def make_puzzle(solution, givens, seed):
    rng = random.Random(seed)
    n = len(solution)
    total = n * n
    keep = set(rng.sample(range(total), givens))
    puzzle = []
    for i in range(n):
        row = []
        for j in range(n):
            idx = i * n + j
            row.append(solution[i][j] if idx in keep else 0)
        puzzle.append(row)
    return puzzle

def write_board(path, p, q, board):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{p} {q}\n")
        for row in board:
            f.write(" ".join(str(x) for x in row) + "\n")

def make_unsolvable_from(board):
    bad = [row[:] for row in board]
    n = len(bad)
    # Force a row contradiction by duplicating a non-zero value.
    # Prefer modifying a blank; if there is none, overwrite a different cell.
    row = 0
    existing = next((x for x in bad[row] if x != 0), None)
    if existing is None:
        existing = 1
    target_col = next((j for j, x in enumerate(bad[row]) if x == 0), None)
    if target_col is None:
        target_col = 1 if n > 1 else 0
    bad[row][target_col] = existing
    return bad

for name, p, q, givens in settings:
    out_dir = boards_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(boards_per):
        sol = make_solution(p, q, seed=10_000 * (p * 100 + q) + i)
        puz = make_puzzle(sol, givens=givens, seed=20_000 * (p * 100 + q) + i)
        write_board(out_dir / f"board_{i}.txt", p, q, puz)

# One unsolvable smoke test per difficulty.
uns_dir = boards_dir / "unsolvable"
uns_dir.mkdir(parents=True, exist_ok=True)
for name, p, q, givens in settings:
    sol = make_solution(p, q, seed=999_000 + p * 100 + q)
    puz = make_puzzle(sol, givens=givens, seed=777_000 + p * 100 + q)
    bad = make_unsolvable_from(puz)
    write_board(uns_dir / f"{name}_unsolvable.txt", p, q, bad)
PY

run_one() {
  local board_file="$1"
  local json_out="$2"
  local timeout_secs="$3"

  if timeout "$timeout_secs"s python3 - "$SRC_DIR" "$board_file" "$json_out" <<'PY' >/dev/null 2>&1
import json
import os
import sys
from pathlib import Path

src_dir = sys.argv[1]
board_file = sys.argv[2]
json_out = sys.argv[3]

sys.path.insert(0, src_dir)

import SudokuBoard
import BTSolver
import Trail


def validate_solution(original_board, solved_board):
    p = original_board.p
    q = original_board.q
    n = p * q
    orig = original_board.board
    sol = solved_board.board

    if len(sol) != n or any(len(row) != n for row in sol):
        return False, "wrong_shape"

    symbols = set(range(1, n + 1))

    # preserve givens + no zeroes
    for i in range(n):
        for j in range(n):
            if sol[i][j] == 0:
                return False, "contains_zero"
            if orig[i][j] != 0 and orig[i][j] != sol[i][j]:
                return False, "changed_given"

    # rows
    for i in range(n):
        if set(sol[i]) != symbols:
            return False, f"bad_row_{i}"

    # cols
    for j in range(n):
        col = {sol[i][j] for i in range(n)}
        if col != symbols:
            return False, f"bad_col_{j}"

    # blocks
    for br in range(0, n, p):
        for bc in range(0, n, q):
            vals = []
            for i in range(br, br + p):
                for j in range(bc, bc + q):
                    vals.append(sol[i][j])
            if set(vals) != symbols:
                return False, f"bad_block_{br}_{bc}"

    return True, "ok"

try:
    board = SudokuBoard.SudokuBoard(filepath=board_file)
    trail = Trail.Trail()
    solver = BTSolver.BTSolver(board, trail, "LeastConstrainingValue", "MinimumRemainingValue", "forwardChecking")

    pre_ok = solver.checkConsistency()
    timed_out_from_solver = False
    if pre_ok:
        rc = solver.solve()
        timed_out_from_solver = (rc == -1)

    result = {
        "board": board_file,
        "pre_consistent": bool(pre_ok),
        "solver_timeout_flag": bool(timed_out_from_solver),
        "solved": bool(solver.hassolution),
        "backtracks": int(trail.getUndoCount()),
        "trail_pushes": int(trail.getPushCount()),
        "valid_solution": False,
        "validation_reason": None,
    }

    if solver.hassolution:
        solved = solver.getSolution()
        ok, reason = validate_solution(board, solved)
        result["valid_solution"] = bool(ok)
        result["validation_reason"] = reason
    else:
        result["validation_reason"] = "no_solution"

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(result, f)
except Exception as e:
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({
            "board": board_file,
            "error": type(e).__name__,
            "message": str(e)
        }, f)
PY
  then
    return 0
  else
    printf '{"board": "%s", "timeout": true}\n' "$board_file" > "$json_out"
    return 124
  fi
}

echo "[STEP] Running Draft AI test suite..."

audit_set() {
  local label="$1"
  local dir="$2"
  local timeout_secs="$3"
  local threshold_num="$4"
  local threshold_den="$5"
  local expect_unsolved_ok="$6"

  local out_jsonl="$RESULTS_DIR/${label}.jsonl"
  : > "$out_jsonl"

  local total=0
  for board in "$dir"/*.txt; do
    [[ -e "$board" ]] || continue
    total=$((total + 1))
    local stem
    stem="$(basename "$board" .txt)"
    local tmp_json="$RESULTS_DIR/${label}_${stem}.json"
    run_one "$board" "$tmp_json" "$timeout_secs" || true
    cat "$tmp_json" >> "$out_jsonl"
    printf '\n' >> "$out_jsonl"
  done

  python3 - "$label" "$out_jsonl" "$threshold_num" "$threshold_den" "$expect_unsolved_ok" <<'PY'
import json
import sys
from pathlib import Path

label = sys.argv[1]
jsonl = Path(sys.argv[2])
threshold_num = int(sys.argv[3])
threshold_den = int(sys.argv[4])
expect_unsolved_ok = sys.argv[5] == "1"

rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]

total = len(rows)
solved_valid = 0
invalid = 0
timeouts = 0
errors = 0
unsolved = 0
pre_inconsistent = 0

for r in rows:
    if r.get("timeout"):
        timeouts += 1
        continue
    if "error" in r:
        errors += 1
        continue
    if not r.get("pre_consistent", True):
        pre_inconsistent += 1
    if r.get("solved") and r.get("valid_solution"):
        solved_valid += 1
    elif r.get("solved") and not r.get("valid_solution"):
        invalid += 1
    else:
        unsolved += 1

passes_threshold = (solved_valid * threshold_den >= threshold_num * total) if total else False

print(f"===== {label.upper()} =====")
print(f"total boards        : {total}")
print(f"solved + valid      : {solved_valid}")
print(f"unsolved            : {unsolved}")
print(f"invalid solutions   : {invalid}")
print(f"timeouts            : {timeouts}")
print(f"runtime errors      : {errors}")
print(f"pre-inconsistent    : {pre_inconsistent}")

if not expect_unsolved_ok:
    pct = (100.0 * solved_valid / total) if total else 0.0
    need = 100.0 * threshold_num / threshold_den
    print(f"completion rate     : {pct:.1f}%")
    print(f"required threshold  : {need:.1f}%")
    print(f"threshold pass      : {'YES' if passes_threshold else 'NO'}")
    # Official score cannot be exactly computed locally.
    print("score check         : completion threshold checked locally; official average-score check depends on hidden teacher backtrack baseline")
else:
    ok = (invalid == 0 and errors == 0 and timeouts == 0 and solved_valid == 0)
    print(f"expected outcome    : unsolved / rejected")
    print(f"unsolvable smoke    : {'PASS' if ok else 'CHECK'}")

print()
PY
}

audit_set "easy" "$BOARDS_DIR/easy" "$TIMEOUT_EASY" 70 100 0
audit_set "intermediate" "$BOARDS_DIR/intermediate" "$TIMEOUT_INTERMEDIATE" 50 100 0
audit_set "hard" "$BOARDS_DIR/hard" "$TIMEOUT_HARD" 30 100 0
audit_set "unsolvable" "$BOARDS_DIR/unsolvable" "$TIMEOUT_UNSOLVABLE" 0 1 1

python3 - "$RESULTS_DIR/easy.jsonl" "$RESULTS_DIR/intermediate.jsonl" "$RESULTS_DIR/hard.jsonl" <<'PY'
import json
import sys
from pathlib import Path

def solved_valid_count(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(rows)
    good = 0
    for r in rows:
        if r.get("timeout") or "error" in r:
            continue
        if r.get("solved") and r.get("valid_solution"):
            good += 1
    return good, total

thresholds = {
    "easy": (70, 100),
    "intermediate": (50, 100),
    "hard": (30, 100),
}

summary = {}
for label, path in zip(["easy", "intermediate", "hard"], sys.argv[1:]):
    good, total = solved_valid_count(path)
    num, den = thresholds[label]
    summary[label] = (good, total, good * den >= num * total if total else False)

all_pass = all(v[2] for v in summary.values())
print("===== OVERALL DRAFT AI CHECK =====")
for label in ["easy", "intermediate", "hard"]:
    good, total, ok = summary[label]
    pct = 100.0 * good / total if total else 0.0
    print(f"{label:13s}: {good:3d}/{total:<3d} solved-valid ({pct:5.1f}%) -> {'PASS' if ok else 'FAIL'}")
print(f"overall result : {'PASS' if all_pass else 'FAIL'}")
print()
print("Artifacts:")
print(f"  Raw per-board JSON results are in: {Path(sys.argv[1]).parent}")
PY

echo "[DONE] Draft AI test run complete."
echo "       Raw outputs are under: $RESULTS_DIR"

