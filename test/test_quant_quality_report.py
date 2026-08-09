#!/usr/bin/env python3
"""GPU-free integration test for the quant report generator."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "quant_quality_report.py"


SCORE_HEADER = (
    "id\tprompt_tokens\ttarget_tokens\tnll\tavg_nll\tfirst_match\tgreedy_lcp"
    "\tapi_target_tokens\tapi_target_mae\tapi_target_mean_delta"
    "\tapi_top_items\tapi_top_mapped\tapi_top1_count\tapi_top1_match"
    "\tapi_topn_ref\tapi_topn_hit\tapi_top_logprob_count\tapi_top_mae"
    "\tapi_top_mean_delta\tapi_pair_total\tapi_pair_agree\n"
)


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ember-quant-report-") as tmp:
        root = Path(tmp)
        metadata = {
            "title": "Synthetic quant report",
            "reference": {"name": "source", "artifact": "source.gguf", "sha256": "abc"},
            "quant": {"name": "quant", "artifact": "quant.gguf", "sha256": "def", "importance_matrix": "imatrix.dat", "importance_matrix_sha256": "987"},
            "evaluation": {"context_tokens": 64, "chunks": 8},
            "gates": [
                {
                    "label": "NLL delta",
                    "metric": "scores.comparison.avg_nll_delta_quant_minus_reference",
                    "op": "<=",
                    "value": 0.3,
                },
                {
                    "label": "Behavior pass",
                    "metric": "behavior.quant.success_rate",
                    "op": ">=",
                    "value": 0.5,
                },
            ],
        }
        write(root / "metadata.json", json.dumps(metadata))
        write(
            root / "reference.tsv",
            SCORE_HEADER
            + "a\t4\t2\t2.0\t1.0\t1\t2\t2\t0.1\t0\t4\t4\t2\t2\t4\t3\t4\t0.2\t0\t6\t5\n"
            + "b\t4\t1\t2.0\t2.0\t0\t0\t1\t0.2\t0\t2\t2\t1\t0\t2\t1\t2\t0.3\t0\t1\t1\n",
        )
        write(
            root / "quant.tsv",
            SCORE_HEADER
            + "a\t4\t2\t2.4\t1.2\t1\t1\t2\t0.2\t0\t4\t4\t2\t1\t4\t2\t4\t0.3\t0\t6\t4\n"
            + "b\t4\t1\t2.3\t2.3\t0\t0\t1\t0.3\t0\t2\t2\t1\t0\t2\t1\t2\t0.4\t0\t1\t1\n",
        )
        write(
            root / "behavior.jsonl",
            "\n".join(
                json.dumps(row)
                for row in (
                    {"variant": "reference", "case_id": "x", "category": "tool", "success": True, "tool_call_valid": True, "repetition_detected": False},
                    {"variant": "quant", "case_id": "x", "category": "tool", "success": True, "tool_call_valid": True, "repetition_detected": False},
                    {"variant": "quant", "case_id": "y", "category": "code", "success": False, "tool_call_valid": None, "repetition_detected": True},
                )
            )
            + "\n",
        )
        write(
            root / "distribution.log",
            "| Mean PPL(Q)                    |      6.227711 +/- 0.037833 |\n"
            "| Mean PPL(base)                 |      6.225194 +/- 0.037771 |\n"
            "| Mean    KLD                    |      0.120000 +/- 0.001000 |\n"
            "| Maximum KLD                    |                 2.500000 |\n"
            "| 99.9%   KLD                    |                 1.250000 |\n"
            "| Mean    Δp                     | -1.250 +/- 0.100 % |\n"
            "| 99.9%   Δp                     | 37.184% |\n"
            "| RMS Δp                         | 13.400 +/- 0.100 % |\n"
            "| Same top p                     | 84.500 +/- 0.100 % |\n",
        )
        write(
            root / "runtime.json",
            json.dumps(
                {
                    "reference": {"artifact_bytes": 1000, "peak_rss_bytes": 2000, "prefill_tokens_per_second": 4, "decode_tokens_per_second": 2},
                    "quant": {"artifact_bytes": 500, "peak_rss_bytes": 1200, "prefill_tokens_per_second": 8, "decode_tokens_per_second": 3},
                }
            ),
        )
        write(
            root / "structural.json",
            json.dumps(
                {
                    "complete": True,
                    "version": 3,
                    "tensors": 20,
                    "actual_file_bytes": 500,
                    "expected_file_bytes": 500,
                    "trailing_bytes": 0,
                    "type_counts": {"q8_0": 10, "f16": 10},
                }
            ),
        )
        write(
            root / "tensor.tsv",
            "tensor\tclass\tnrmse\tweighted_nrmse\tcosine\n"
            "blk.0.ffn_gate_exps\trouted_gate\t0.3\t0.2\t0.9\n"
            "blk.1.ffn_gate_exps\trouted_gate\t0.4\t0.3\t0.8\n",
        )
        write(
            root / "variants.tsv",
            "name\tsize_gib\tbpw\tmean_kld\tp99_9_kld\tmaximum_kld\tsame_top_probability_pct\tbehavioral_pass_rate\n"
            "BF16\t150\t8\t0.001\t0.01\t0.1\t99.7\t1.0\n"
            "ROCMFP2\t90\t2.7\t0.12\t1.25\t2.5\t84.5\t0.5\n",
        )
        write(
            root / "modes.tsv",
            "name\tartifact\tprefill\tdrafter\ttool_result_guard\tcases\tbehavioral_pass_rate\ttool_result_pass_rate\trepetition_detected_rate\tidentifier_integrity_rate\tprefill_tokens_per_second\tdecode_tokens_per_second\n"
            "exact AR\tROCMFP2\texact\toff\ton\t15\t0.9333\t1\t0\t0\t17.7\t18.3\n"
            "sparse spec\tROCMFP2\tsparse\ton\toff\t15\t0.8667\t0.667\t0.0667\t0\t120\t25\n",
        )
        out = root / "report"
        command = [
            sys.executable,
            str(SCRIPT),
            "--metadata", str(root / "metadata.json"),
            "--reference-scores", str(root / "reference.tsv"),
            "--quant-scores", str(root / "quant.tsv"),
            "--distribution", str(root / "distribution.log"),
            "--behavior", str(root / "behavior.jsonl"),
            "--runtime", str(root / "runtime.json"),
            "--structural-audit", str(root / "structural.json"),
            "--tensor-error", str(root / "tensor.tsv"),
            "--variants", str(root / "variants.tsv"),
            "--modes", str(root / "modes.tsv"),
            "--out-dir", str(out),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)
            return 1

        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        expected_delta = (4.7 - 4.0) / 3
        actual_delta = report["scores"]["comparison"]["avg_nll_delta_quant_minus_reference"]
        assert math.isclose(actual_delta, expected_delta, rel_tol=1e-9)
        assert report["gates"]["status"] == "pass"
        assert report["distribution"]["mean_kld"] == 0.12
        assert report["distribution"]["same_top_probability_pct"] == 84.5
        assert report["distribution"]["p99_9_kld"] == 1.25
        assert report["distribution"]["reference_perplexity"] == 6.225194
        assert report["distribution"]["quant_perplexity"] == 6.227711
        assert report["distribution"]["delta_probability_percentiles_pct"]["99.9"] == 37.184
        markdown = (out / "report.md").read_text(encoding="utf-8")
        assert "Source perplexity" in markdown
        assert "## Artifacts" in markdown
        assert "Importance matrix: `imatrix.dat` (SHA-256 `987`)" in markdown
        assert "Context Tokens: 64" in markdown
        assert "## Sampled tensor reconstruction" in markdown
        assert "## GGUF structural audit" in markdown
        assert "Raw DSML" in markdown
        assert report["behavior"]["quant"]["success_rate"] == 0.5
        assert report["tensor_error"]["classes"]["routed_gate"]["mean_weighted_nrmse"] == 0.25
        assert report["artifacts"]["plots"]
        assert len(report["variants"]) == 2
        assert markdown.startswith("# Synthetic quant report")
        assert (out / "nll_delta_by_case.svg").read_text(encoding="utf-8").startswith("<svg")
        assert (out / "size_vs_kld.svg").is_file()
        assert (out / "size_vs_p99_9_kld.svg").is_file()
        assert (out / "runtime_footprint.svg").is_file()
        assert (out / "operating_mode_safety.svg").is_file()
        assert (out / "full_logit_divergence.svg").is_file()
        assert (out / "full_logit_probability.svg").is_file()
        for plot in report["artifacts"]["plots"]:
            ET.parse(out / plot)
        assert (out / "case_metrics.csv").is_file()
        assert (out / "behavior_cases.csv").is_file()
        assert len(report["behavior_cases"]) == 3

        write(root / "metadata-behavior.json", json.dumps({"title": "Behavior only"}))
        behavior_out = root / "behavior-report"
        behavior_command = [
            sys.executable,
            str(SCRIPT),
            "--metadata", str(root / "metadata-behavior.json"),
            "--behavior", str(root / "behavior.jsonl"),
            "--out-dir", str(behavior_out),
        ]
        completed = subprocess.run(behavior_command, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        behavior_markdown = (behavior_out / "report.md").read_text(encoding="utf-8")
        assert "candidate versus 100.0% for the upstream endpoint" in behavior_markdown

        write(root / "metadata-ungated.json", json.dumps({"title": "Distribution-only"}))
        distribution_out = root / "distribution-report"
        distribution_command = [
            sys.executable,
            str(SCRIPT),
            "--metadata", str(root / "metadata-ungated.json"),
            "--distribution", str(root / "distribution.log"),
            "--behavior", str(root / "behavior.jsonl"),
            "--runtime", str(root / "runtime.json"),
            "--structural-audit", str(root / "structural.json"),
            "--tensor-error", str(root / "tensor.tsv"),
            "--variants", str(root / "variants.tsv"),
            "--modes", str(root / "modes.tsv"),
            "--out-dir", str(distribution_out),
        ]
        completed = subprocess.run(distribution_command, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        distribution_markdown = (distribution_out / "report.md").read_text(encoding="utf-8")
        assert "mean full-vocabulary KL divergence was 0.120000" in distribution_markdown
        assert not (distribution_out / "case_metrics.csv").exists()
        distribution_report = json.loads(
            (distribution_out / "report.json").read_text(encoding="utf-8")
        )
        assert distribution_report["gates"]["status"] == "not_gated"
    print("quant quality report: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
