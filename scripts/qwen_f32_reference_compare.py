#!/usr/bin/env python3
"""Offline comparison for the Qwen F32 dequantized-reference run.

Consumes the raw little-endian F32 logit rows written by
``EMBER_VALIDATION_LOGITS_DIR`` and reports, per width:

    d_q1   = max |q1        - reference|
    d_prod = max |production - reference|

``q1`` and ``production`` come from the DEFAULT build; ``reference`` is the
production stream of the ``GGML_CUDA_FORCE_CUBLAS`` +
``DFLASH_CUBLAS_F32_REFERENCE=1`` build.

Width 2 gates the run.  ``sync_fallback`` is recorded in
``docs/dead-code-candidates.md`` entry 3 as absent from measured production
dispatches, and the reference build routes every routed expert through it, so
the reference is computed on code that production never exercises.  At width
2 the default production stream is bit-identical to q1, so any distance
measured there is the REFERENCE's own error and nothing else.  If that distance
is not small, the finding is about ``sync_fallback`` and says nothing about
MMQ.  The measurements supporting those statements live only in
``docs/qwen3.8-performance-status.md``.

This script decides nothing on its own: it prints the numbers and the gate
verdict.  Exit status is 0 when the comparison is well-formed, 1 when a
structural contract is violated (missing/short/ragged/non-finite rows), which
is a voided run rather than a result.
"""

import argparse
import math
import pathlib
import struct
import sys

GATE_WIDTH = 2


def read_rows(directory: pathlib.Path, stem: str) -> list[list[float]]:
    """Read ``<stem>-rowNNN.f32`` files in index order."""
    paths = sorted(directory.glob(f"{stem}-row*.f32"))
    if not paths:
        raise ValueError(f"no {stem}-row*.f32 files in {directory}")
    rows = []
    for path in paths:
        raw = path.read_bytes()
        if len(raw) % 4:
            raise ValueError(f"{path} is {len(raw)} bytes, not a whole number of F32")
        rows.append(list(struct.unpack(f"<{len(raw) // 4}f", raw)))
    return rows


def max_abs_delta(left: list[float], right: list[float]) -> float:
    worst = 0.0
    for a, b in zip(left, right):
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError("non-finite logit encountered")
        worst = max(worst, abs(a - b))
    return worst


def rank_report(authority: list[float], candidate: list[float],
                windows: tuple[int, ...] = (1, 2, 10, 50)) -> dict:
    """Rank-aware view of a single row's disagreement.

    ``authority`` must be the TRUSTED side, because ranking decides which
    disagreements can matter and only the trusted distribution says which
    tokens are plausible. Passing the untrusted side inverts the question: it
    reports whether the broken path's own favourites were preserved, which is
    not what a decision depends on.

    ``max_abs_delta`` over a full vocabulary is rank-blind: it cannot tell an
    error on the second-most-likely token from one on a token ranked a quarter
    million deep, which cannot affect sampling at any temperature we ship.
    Restricting the same statistic to the reference's own top-ranked logits
    says whether a disagreement can change the decision or only the noise
    floor.  ``worst_rank`` locates the global maximum in that ordering.
    """
    order = sorted(range(len(authority)), key=lambda i: -authority[i])
    deltas = [abs(a - b) for a, b in zip(authority, candidate)]
    worst = max(range(len(deltas)), key=lambda i: deltas[i])
    top1, top2 = order[0], order[1]
    return {
        "windows": {k: max(deltas[i] for i in order[:k]) for k in windows
                    if k <= len(order)},
        "worst_rank": order.index(worst),
        "vocab": len(authority),
        "argmax_agrees":
            max(range(len(candidate)), key=lambda i: candidate[i]) == top1,
        "margin_authority": authority[top1] - authority[top2],
        "margin_candidate": candidate[top1] - candidate[top2],
    }


def compare(default_dir: pathlib.Path, reference_dir: pathlib.Path) -> tuple[float, float, int]:
    q1 = read_rows(default_dir, "q1")
    production = read_rows(default_dir, "production")
    reference = read_rows(reference_dir, "production")

    counts = {len(q1), len(production), len(reference)}
    if len(counts) != 1:
        raise ValueError(
            f"row counts differ: q1={len(q1)} production={len(production)} "
            f"reference={len(reference)}"
        )
    for index, (a, b, c) in enumerate(zip(q1, production, reference)):
        if not (len(a) == len(b) == len(c)):
            raise ValueError(
                f"row {index} widths differ: {len(a)}, {len(b)}, {len(c)}"
            )

    d_q1 = max(max_abs_delta(a, c) for a, c in zip(q1, reference))
    d_prod = max(max_abs_delta(b, c) for b, c in zip(production, reference))
    return d_q1, d_prod, len(q1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--width", action="append", required=True, metavar="W:DEFAULT_DIR:REFERENCE_DIR",
                        help="repeatable, e.g. 2:/ev/w2/default:/ev/w2/reference")
    parser.add_argument("--ranks", action="store_true",
                        help="also print the rank-aware breakdown per row: max "
                             "delta restricted to the top-1/2/10/50 logits, the "
                             "rank of the worst deviation, argmax agreement, and "
                             "the top-2 margin on both sides")
    parser.add_argument("--gate-ratio", type=float, default=10.0,
                        help="the reference's own error must be at least this many "
                             "times SMALLER than the smallest effect it has to "
                             "resolve (default 10.0)")
    args = parser.parse_args()

    results = {}
    dirs = {}
    for spec in args.width:
        try:
            width_text, default_text, reference_text = spec.split(":", 2)
            width = int(width_text)
        except ValueError:
            print(f"malformed --width {spec!r}, want W:DEFAULT_DIR:REFERENCE_DIR", file=sys.stderr)
            return 1
        try:
            d_q1, d_prod, rows = compare(pathlib.Path(default_text), pathlib.Path(reference_text))
        except ValueError as error:
            print(f"width {width}: VOID — {error}", file=sys.stderr)
            return 1
        results[width] = (d_q1, d_prod, rows)
        dirs[width] = (pathlib.Path(default_text), pathlib.Path(reference_text))

    print(f"{'width':>6} {'rows':>5} {'d_q1':>14} {'d_prod':>14} {'d_prod/d_q1':>13}")
    for width in sorted(results):
        d_q1, d_prod, rows = results[width]
        ratio = f"{d_prod / d_q1:.4g}" if d_q1 > 0.0 else "n/a"
        print(f"{width:>6} {rows:>5} {d_q1:>14.6g} {d_prod:>14.6g} {ratio:>13}")

    if args.ranks:
        print(f"\n{'width':>6} {'row':>4} {'top-1':>9} {'top-2':>9} {'top-10':>9} "
              f"{'top-50':>9} {'worst rank':>11} {'argmax':>7} "
              f"{'margin trust':>12} {'margin cand':>12}")
        for width in sorted(results):
            default_dir, reference_dir = dirs[width]
            production = read_rows(default_dir, "production")
            reference = read_rows(reference_dir, "production")
            # production (default build) is the authority; the F32 reference
            # build is the candidate under test.
            for index, (act, ref) in enumerate(zip(production, reference)):
                r = rank_report(act, ref)
                w = r["windows"]
                print(f"{width:>6} {index:>4} "
                      f"{w.get(1, float('nan')):>9.4f} {w.get(2, float('nan')):>9.4f} "
                      f"{w.get(10, float('nan')):>9.4f} {w.get(50, float('nan')):>9.4f} "
                      f"{r['worst_rank']:>11} {str(r['argmax_agrees']):>7} "
                      f"{r['margin_authority']:>12.4f} "
                      f"{r['margin_candidate']:>12.4f}")
        print(f"\n  'worst rank' is the position of the largest deviation in the "
              f"TRUSTED side's ordering.\n  A rank far from 0 means the headline "
              f"max delta is set by a token that cannot be sampled.")

    if GATE_WIDTH not in results:
        print(f"\nGATE: NOT EVALUATED — width {GATE_WIDTH} absent. The reference's own "
              f"error is unmeasured, so nothing here may be read as evidence about MMQ.")
        return 0

    gate_d_prod = results[GATE_WIDTH][1]
    others = [d for w, (_, d, _) in results.items() if w != GATE_WIDTH and d > 0.0]
    print(f"\nGATE (width {GATE_WIDTH}): reference's own error d_prod = {gate_d_prod:.6g}")
    if not others:
        print("  no other widths to compare against; report the number, draw no conclusion.")
        return 0
    # The reference must be substantially MORE accurate than the effect under
    # test. Allowing its error merely to be "no larger" would let a reference
    # with error 11 adjudicate a difference of 12, which resolves nothing.
    smallest_effect = min(others)
    budget = smallest_effect / args.gate_ratio
    if gate_d_prod <= budget:
        print(f"  PASS — at least {args.gate_ratio}x smaller than the smallest effect "
              f"under test ({smallest_effect:.6g}); budget was {budget:.6g}.")
        print("  The reference is trustworthy enough to interpret the other widths.")
    else:
        print(f"  FAIL — not {args.gate_ratio}x smaller than the smallest effect under "
              f"test ({smallest_effect:.6g}); budget was {budget:.6g}.")
        print("  The finding is about sync_fallback, NOT about MMQ. Do not report this")
        print("  run as evidence for or against the quantized matmul path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
