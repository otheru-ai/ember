#!/usr/bin/env python3
"""Build an auditable Markdown/JSON/SVG report for a model quantization.

The primary score format is the TSV emitted by Dwarfstar's
gguf-tools/quality-testing/score_official.  Optional inputs add full-logit
distribution metrics, behavioral replays, runtime measurements, and sampled
weight-reconstruction error.  The script deliberately has no third-party
dependencies so reports can be generated on an offline build host.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import math
import platform
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


SCHEMA_VERSION = 1


def die(message: str) -> "NoReturn":
    raise SystemExit(message)


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_provenance(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader((line for line in fp if not line.startswith("#")), delimiter="\t"))
    if not rows:
        die(f"{path}: no data rows")
    return rows


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                die(f"{path}:{line_no}: {exc}")
            if not isinstance(row, dict):
                die(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    preferred = ["variant", "case_id", "category", "success"]
    keys = {key for row in rows for key in row}
    fields = [key for key in preferred if key in keys] + sorted(keys - set(preferred))
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def score_aggregate(rows: list[dict[str, str]]) -> dict[str, Any]:
    tokens = sum(integer(row.get("target_tokens")) for row in rows)
    total_nll = sum(finite(row.get("nll")) or 0.0 for row in rows)
    cases = len(rows)
    avg_nll = total_nll / tokens if tokens else None

    result: dict[str, Any] = {
        "cases": cases,
        "target_tokens": tokens,
        "avg_nll": avg_nll,
        "perplexity": math.exp(avg_nll) if avg_nll is not None and avg_nll < 700 else None,
        "official_first_token_match_rate": (
            sum(integer(row.get("first_match")) for row in rows) / cases if cases else None
        ),
        "avg_greedy_lcp": (
            sum(integer(row.get("greedy_lcp")) for row in rows) / cases if cases else None
        ),
    }

    weighted = {
        "api_target_mae": "api_target_tokens",
        "api_target_mean_delta": "api_target_tokens",
        "api_top_mae": "api_top_logprob_count",
        "api_top_mean_delta": "api_top_logprob_count",
    }
    for value_key, count_key in weighted.items():
        count = sum(integer(row.get(count_key)) for row in rows)
        total = sum(
            (finite(row.get(value_key)) or 0.0) * integer(row.get(count_key)) for row in rows
        )
        result[value_key] = total / count if count else None
        result[f"{value_key}_tokens"] = count

    ratios = {
        "api_top_coverage": ("api_top_mapped", "api_top_items"),
        "api_top1_rate": ("api_top1_match", "api_top1_count"),
        "api_topn_recall": ("api_topn_hit", "api_topn_ref"),
        "api_pair_rate": ("api_pair_agree", "api_pair_total"),
    }
    for key, (num_key, den_key) in ratios.items():
        num = sum(integer(row.get(num_key)) for row in rows)
        den = sum(integer(row.get(den_key)) for row in rows)
        result[key] = num / den if den else None
        result[f"{key}_count"] = den
    return result


def compare_scores(
    reference_rows: list[dict[str, str]], quant_rows: list[dict[str, str]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ref = {row.get("id", ""): row for row in reference_rows}
    quant = {row.get("id", ""): row for row in quant_rows}
    ids = sorted(set(ref) & set(quant))
    if not ids:
        die("reference and quant score files have no common case IDs")
    missing_reference = sorted(set(quant) - set(ref))
    missing_quant = sorted(set(ref) - set(quant))
    per_case: list[dict[str, Any]] = []
    wins = {"reference": 0, "quant": 0, "ties": 0}
    for case_id in ids:
        rr, qr = ref[case_id], quant[case_id]
        rt, qt = integer(rr.get("target_tokens")), integer(qr.get("target_tokens"))
        if rt != qt:
            die(f"target-token count mismatch for {case_id}: reference={rt}, quant={qt}")
        ravg, qavg = finite(rr.get("avg_nll")), finite(qr.get("avg_nll"))
        if ravg is None or qavg is None:
            die(f"missing avg_nll for {case_id}")
        delta = qavg - ravg
        if delta < -1e-12:
            wins["quant"] += 1
        elif delta > 1e-12:
            wins["reference"] += 1
        else:
            wins["ties"] += 1
        per_case.append(
            {
                "id": case_id,
                "target_tokens": rt,
                "reference_avg_nll": ravg,
                "quant_avg_nll": qavg,
                "delta_avg_nll": delta,
                "reference_first_match": bool(integer(rr.get("first_match"))),
                "quant_first_match": bool(integer(qr.get("first_match"))),
                "reference_greedy_lcp": integer(rr.get("greedy_lcp")),
                "quant_greedy_lcp": integer(qr.get("greedy_lcp")),
                "category": qr.get("category") or rr.get("category") or "all",
            }
        )

    ref_common = [ref[case_id] for case_id in ids]
    quant_common = [quant[case_id] for case_id in ids]
    ra, qa = score_aggregate(ref_common), score_aggregate(quant_common)
    delta_nll = (qa["avg_nll"] - ra["avg_nll"]) if ra["avg_nll"] is not None else None
    comparison = {
        "common_cases": len(ids),
        "missing_from_reference": missing_reference,
        "missing_from_quant": missing_quant,
        "case_wins": wins,
        "avg_nll_delta_quant_minus_reference": delta_nll,
        "perplexity_ratio_quant_over_reference": (
            math.exp(delta_nll) if delta_nll is not None and delta_nll < 700 else None
        ),
        "official_first_token_match_rate_delta": (
            qa["official_first_token_match_rate"] - ra["official_first_token_match_rate"]
        ),
        "avg_greedy_lcp_delta": qa["avg_greedy_lcp"] - ra["avg_greedy_lcp"],
    }
    return {"reference": ra, "quant": qa, "comparison": comparison}, per_case


def parse_distribution(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        data = load_json(path)
        if not isinstance(data, dict):
            die(f"{path}: distribution JSON must be an object")
        return data

    text = path.read_text(encoding="utf-8", errors="replace")
    row = r"^\s*\|?\s*"
    separator = r"\s*(?:\||:)\s*"
    number = r"([-+0-9.eE]+)"
    patterns = {
        "mean_kld": row + r"Mean\s+(?:KL(?:\s+divergence)?|KLD)" + separator + number,
        "maximum_kld": row + r"Maximum\s+KLD" + separator + number,
        "p99_9_kld": row + r"99\.9%\s+KLD" + separator + number,
        "p99_kld": row + r"99\.0?%\s+KLD" + separator + number,
        "same_top_probability_pct": row + r"Same\s+top\s+p" + separator + number,
        "mean_delta_probability_pct": row + r"Mean\s+(?:Δ|delta\s*)p" + separator + number,
        "rms_delta_probability_pct": row + r"RMS\s+(?:Δ|delta\s*)p" + separator + number,
        "reference_perplexity": row + r"Mean\s+PPL\(base\)" + separator + number,
        "quant_perplexity": (
            r"(?:Final estimate:\s*PPL\s*=\s*|"
            + row
            + r"Mean\s+PPL\(Q\)"
            + separator
            + r")"
            + number
        ),
    }
    result: dict[str, Any] = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            result[key] = float(matches[-1])

    percentile_patterns = {
        "99.9": row + r"99\.9%\s+(?:Δ|delta\s*)p" + separator + number,
        "99": row + r"99\.0?%\s+(?:Δ|delta\s*)p" + separator + number,
        "95": row + r"95\.0?%\s+(?:Δ|delta\s*)p" + separator + number,
        "50": row + r"Median\s+(?:Δ|delta\s*)p" + separator + number,
        "5": row + r"5\.0?%\s+(?:Δ|delta\s*)p" + separator + number,
        "1": row + r"1\.0?%\s+(?:Δ|delta\s*)p" + separator + number,
        "0.1": row + r"0\.1%\s+(?:Δ|delta\s*)p" + separator + number,
    }
    percentiles = {}
    for key, pattern in percentile_patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            percentiles[key] = float(match.group(1))
    if percentiles:
        result["delta_probability_percentiles_pct"] = percentiles
    if not result:
        die(f"{path}: no recognized llama-perplexity metrics")
    return result


BEHAVIOR_FIELDS = {
    "success": True,
    "response_valid": True,
    "tool_call_valid": True,
    "identifier_integrity": True,
    "repetition_detected": False,
    "dsml_leak_detected": False,
    "thinking_leak_detected": False,
}


def behavior_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row.get("variant", "quant"))].append(row)
    result: dict[str, Any] = {}
    for variant, items in sorted(by_variant.items()):
        aggregate: dict[str, Any] = {"cases": len(items)}
        for field in BEHAVIOR_FIELDS:
            values = [bool(row[field]) for row in items if field in row and row[field] is not None]
            aggregate[f"{field}_rate"] = sum(values) / len(values) if values else None
            aggregate[f"{field}_count"] = len(values)
        for field in (
            "latency_seconds",
            "tokens_per_second",
            "prefill_tokens_per_second",
            "decode_tokens_per_second",
        ):
            values = [value for row in items if (value := finite(row.get(field))) is not None]
            aggregate[f"mean_{field}"] = mean(values) if values else None

        categories: dict[str, dict[str, Any]] = {}
        category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in items:
            category_rows[str(row.get("category", "uncategorized"))].append(row)
        for category, citems in sorted(category_rows.items()):
            successes = [bool(row["success"]) for row in citems if "success" in row]
            categories[category] = {
                "cases": len(citems),
                "success_rate": sum(successes) / len(successes) if successes else None,
            }
        aggregate["categories"] = categories
        result[variant] = aggregate

    if "reference" in result and "quant" in result:
        comparison = {}
        for field in BEHAVIOR_FIELDS:
            key = f"{field}_rate"
            rv, qv = result["reference"].get(key), result["quant"].get(key)
            comparison[f"{field}_rate_delta"] = qv - rv if rv is not None and qv is not None else None
        result["comparison"] = comparison
    return result


def tensor_aggregate(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[row.get("class") or row.get("tensor_class") or "unclassified"].append(row)
    classes: dict[str, Any] = {}
    for name, items in sorted(by_class.items()):
        nrmse = [value for row in items if (value := finite(row.get("nrmse"))) is not None]
        weighted = [
            value for row in items if (value := finite(row.get("weighted_nrmse"))) is not None
        ]
        cosine = [value for row in items if (value := finite(row.get("cosine"))) is not None]
        classes[name] = {
            "tensors": len(items),
            "mean_nrmse": mean(nrmse) if nrmse else None,
            "mean_weighted_nrmse": mean(weighted) if weighted else None,
            "mean_cosine": mean(cosine) if cosine else None,
        }
    return {"tensors": len(rows), "classes": classes}


def load_variants(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        data = load_json(path)
        if isinstance(data, dict):
            data = data.get("variants")
        if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
            die(f"{path}: variants JSON must be a list or an object containing a variants list")
        rows = data
    else:
        rows = load_tsv(path)
    result: list[dict[str, Any]] = []
    numeric_fields = {
        "artifact_bytes",
        "size_gib",
        "bpw",
        "perplexity",
        "delta_perplexity",
        "mean_kld",
        "maximum_kld",
        "p99_9_kld",
        "p99_kld",
        "same_top_probability_pct",
        "mean_delta_probability_pct",
        "rms_delta_probability_pct",
        "behavioral_pass_rate",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
    }
    for index, row in enumerate(rows, 1):
        name = row.get("name")
        if not name:
            die(f"{path}: variant row {index} is missing name")
        normalized: dict[str, Any] = {"name": str(name)}
        for key, value in row.items():
            if key == "name" or value in (None, ""):
                continue
            if key in numeric_fields:
                number = finite(value)
                if number is None:
                    die(f"{path}: variant {name!r} has invalid {key}={value!r}")
                normalized[key] = number
            else:
                normalized[key] = value
        if "size_gib" not in normalized and "artifact_bytes" in normalized:
            normalized["size_gib"] = normalized["artifact_bytes"] / 2**30
        result.append(normalized)
    return result


def load_operating_modes(path: Path) -> list[dict[str, Any]]:
    """Load serving-mode comparisons for one or more model artifacts."""
    if path.suffix.lower() == ".json":
        data = load_json(path)
        if isinstance(data, dict):
            data = data.get("modes")
        if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
            die(f"{path}: modes JSON must be a list or an object containing a modes list")
        rows = data
    else:
        rows = load_tsv(path)
    numeric_fields = {
        "cases",
        "behavioral_pass_rate",
        "tool_result_pass_rate",
        "repetition_detected_rate",
        "identifier_integrity_rate",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
    }
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        name = row.get("name")
        if not name:
            die(f"{path}: mode row {index} is missing name")
        normalized: dict[str, Any] = {"name": str(name)}
        for key, value in row.items():
            if key == "name" or value in (None, ""):
                continue
            if key in numeric_fields:
                number = finite(value)
                if number is None:
                    die(f"{path}: mode {name!r} has invalid {key}={value!r}")
                normalized[key] = number
            else:
                normalized[key] = str(value)
        result.append(normalized)
    return result


def deep_get(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


OPS = {
    "<=": lambda actual, limit: actual <= limit,
    "<": lambda actual, limit: actual < limit,
    ">=": lambda actual, limit: actual >= limit,
    ">": lambda actual, limit: actual > limit,
}


def evaluate_gates(report: dict[str, Any], gates: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for gate in gates:
        metric = str(gate.get("metric", ""))
        op = str(gate.get("op", ""))
        limit = finite(gate.get("value"))
        actual = finite(deep_get(report, metric))
        if not metric or op not in OPS or limit is None:
            die(f"invalid gate: {gate!r}")
        status = "not_evaluated" if actual is None else ("pass" if OPS[op](actual, limit) else "fail")
        results.append(
            {
                "label": gate.get("label") or metric,
                "metric": metric,
                "op": op,
                "limit": limit,
                "actual": actual,
                "status": status,
            }
        )
    if not results:
        status = "not_gated"
    elif any(item["status"] == "fail" for item in results):
        status = "fail"
    elif any(item["status"] == "not_evaluated" for item in results):
        status = "incomplete"
    else:
        status = "pass"
    return {"status": status, "results": results}


def fmt(value: Any, digits: int = 4) -> str:
    number = finite(value)
    return "not measured" if number is None else f"{number:.{digits}f}"


def pct(value: Any, digits: int = 1, signed: bool = False) -> str:
    number = finite(value)
    if number is None:
        return "not measured"
    sign = "+" if signed and number > 0 else ""
    return f"{sign}{number * 100:.{digits}f}%"


def md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def svg_document(title: str, body: str, width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">\n'
        f'<title id="title">{html.escape(title)}</title>\n'
        f'<desc id="desc">{html.escape(title)} generated by quant_quality_report.py</desc>\n'
        '<rect width="100%" height="100%" fill="#0d1117"/>\n'
        '<style>text{font-family:system-ui,sans-serif;fill:#c9d1d9}.axis{stroke:#8b949e;stroke-width:1}'
        '.grid{stroke:#30363d;stroke-width:1}.label{font-size:12px}.small{font-size:10px}'
        '.title{font-size:18px;font-weight:600}</style>\n'
        f'{body}\n</svg>\n'
    )


def write_bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    *,
    percent: bool = False,
) -> None:
    width = 940
    height = max(300, 125 + len(labels) * 34)
    left, right, top, bottom = 210, 40, 62, 54
    plot_w, plot_h = width - left - right, height - top - bottom
    all_values = [value for _, values, _ in series for value in values]
    high = max(all_values + ([1.0] if percent else [0.0]))
    low = min(all_values + [0.0])
    if high == low:
        high = low + 1.0
    body = [f'<text x="24" y="32" class="title">{html.escape(title)}</text>']
    zero_x = left + (0.0 - low) / (high - low) * plot_w
    for tick in range(6):
        value = low + (high - low) * tick / 5
        x = left + plot_w * tick / 5
        tick_label = f"{value * 100:.0f}%" if percent else f"{value:.3g}"
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" class="grid"/>')
        body.append(f'<text x="{x:.1f}" y="{height - 18}" text-anchor="middle" class="small">{tick_label}</text>')
    body.append(f'<line x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{top + plot_h}" class="axis"/>')
    group_h = plot_h / max(1, len(labels))
    bar_h = min(12.0, (group_h - 4) / max(1, len(series)))
    for i, label in enumerate(labels):
        cy = top + (i + 0.5) * group_h
        body.append(f'<text x="{left - 10}" y="{cy + 4:.1f}" text-anchor="end" class="label">{html.escape(label[:30])}</text>')
        for j, (name, values, color) in enumerate(series):
            value = values[i]
            x_value = left + (value - low) / (high - low) * plot_w
            x = min(zero_x, x_value)
            w = max(1.0, abs(x_value - zero_x))
            y = cy - (len(series) * bar_h) / 2 + j * bar_h
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bar_h - 1:.1f}" fill="{color}"/>')
    legend_x = left
    for name, _, color in series:
        body.append(f'<rect x="{legend_x}" y="42" width="10" height="10" fill="{color}"/>')
        body.append(f'<text x="{legend_x + 15}" y="51" class="small">{html.escape(name)}</text>')
        legend_x += 28 + len(name) * 7
    path.write_text(svg_document(title, "\n".join(body), width, height), encoding="utf-8")


def write_case_delta_chart(path: Path, cases: list[dict[str, Any]]) -> None:
    ordered = sorted(cases, key=lambda row: row["delta_avg_nll"], reverse=True)
    labels = [row["id"] for row in ordered]
    values = [row["delta_avg_nll"] for row in ordered]
    # Horizontal bars stay legible for 100 cases; the SVG simply becomes tall.
    write_bar_chart(
        path,
        "Quantization-only NLL delta by continuation (positive is worse)",
        labels,
        [("quant − source", values, "#f85149")],
    )


def write_scatter_chart(
    path: Path,
    title: str,
    points: list[tuple[str, float, float]],
    x_label: str,
    y_label: str,
) -> None:
    width, height = 940, 520
    left, right, top, bottom = 90, 45, 62, 72
    plot_w, plot_h = width - left - right, height - top - bottom
    xs, ys = [point[1] for point in points], [point[2] for point in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xpad = max((xmax - xmin) * 0.08, abs(xmax) * 0.02, 0.1)
    ypad = max((ymax - ymin) * 0.10, abs(ymax) * 0.02, 1e-6)
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = ymin - ypad, ymax + ypad
    body = [f'<text x="24" y="32" class="title">{html.escape(title)}</text>']
    for tick in range(6):
        xv = xmin + (xmax - xmin) * tick / 5
        x = left + plot_w * tick / 5
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" class="grid"/>')
        body.append(f'<text x="{x:.1f}" y="{height - 42}" text-anchor="middle" class="small">{xv:.3g}</text>')
        yv = ymin + (ymax - ymin) * tick / 5
        y = top + plot_h - plot_h * tick / 5
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        body.append(f'<text x="{left - 9}" y="{y + 4:.1f}" text-anchor="end" class="small">{yv:.3g}</text>')
    body += [
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 14}" text-anchor="middle" class="label">{html.escape(x_label)}</text>',
        f'<text x="20" y="{top + plot_h / 2:.1f}" transform="rotate(-90 20 {top + plot_h / 2:.1f})" text-anchor="middle" class="label">{html.escape(y_label)}</text>',
    ]
    colors = ("#58a6ff", "#d2a8ff", "#3fb950", "#ffa657", "#f85149", "#39c5cf")
    for index, (name, xv, yv) in enumerate(points):
        x = left + (xv - xmin) / (xmax - xmin) * plot_w
        y = top + plot_h - (yv - ymin) / (ymax - ymin) * plot_h
        color = colors[index % len(colors)]
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="#f0f6fc" stroke-width="1"/>')
        anchor = "end" if x > left + plot_w * 0.78 else "start"
        dx = -9 if anchor == "end" else 9
        body.append(f'<text x="{x + dx:.1f}" y="{y - 8:.1f}" text-anchor="{anchor}" class="small">{html.escape(name[:36])}</text>')
    path.write_text(svg_document(title, "\n".join(body), width, height), encoding="utf-8")


def make_plots(out: Path, report: dict[str, Any], cases: list[dict[str, Any]]) -> list[str]:
    plots: list[str] = []
    if cases:
        name = "nll_delta_by_case.svg"
        write_case_delta_chart(out / name, cases)
        plots.append(name)

    scores = report.get("scores") or {}
    if scores:
        labels = ["Official first token", "API top-1", "API top-N recall", "API pair ordering"]
        keys = ["official_first_token_match_rate", "api_top1_rate", "api_topn_recall", "api_pair_rate"]
        ref_values = [finite(scores["reference"].get(key)) or 0.0 for key in keys]
        quant_values = [finite(scores["quant"].get(key)) or 0.0 for key in keys]
        name = "output_agreement.svg"
        write_bar_chart(
            out / name,
            "Agreement with frozen upstream continuations",
            labels,
            [("source", ref_values, "#58a6ff"), ("quant", quant_values, "#d2a8ff")],
            percent=True,
        )
        plots.append(name)

    distribution = report.get("distribution") or {}
    divergence_fields = [
        ("Mean", "mean_kld"),
        ("99.9th percentile", "p99_9_kld"),
        ("Maximum", "maximum_kld"),
    ]
    divergence_fields = [
        (label, key)
        for label, key in divergence_fields
        if finite(distribution.get(key)) is not None
    ]
    if divergence_fields:
        name = "full_logit_divergence.svg"
        write_bar_chart(
            out / name,
            "Full-logit KL divergence summary (lower is better)",
            [label for label, _ in divergence_fields],
            [
                (
                    "quant vs source",
                    [float(distribution[key]) for _, key in divergence_fields],
                    "#f85149",
                )
            ],
        )
        plots.append(name)

    probability_fields = [
        ("Same top token", "same_top_probability_pct"),
        ("RMS correct-token Δp", "rms_delta_probability_pct"),
    ]
    probability_fields = [
        (label, key)
        for label, key in probability_fields
        if finite(distribution.get(key)) is not None
    ]
    if probability_fields:
        name = "full_logit_probability.svg"
        write_bar_chart(
            out / name,
            "Full-logit token-probability agreement",
            [label for label, _ in probability_fields],
            [
                (
                    "quant vs source",
                    [float(distribution[key]) / 100.0 for _, key in probability_fields],
                    "#d2a8ff",
                )
            ],
            percent=True,
        )
        plots.append(name)

    behavior = report.get("behavior") or {}
    variants = [name for name in ("reference", "quant") if name in behavior]
    if variants:
        categories = sorted(
            {category for variant in variants for category in behavior[variant].get("categories", {})}
        )
        series = []
        colors = {"reference": "#58a6ff", "quant": "#d2a8ff"}
        for variant in variants:
            values = [
                finite(behavior[variant].get("categories", {}).get(category, {}).get("success_rate"))
                or 0.0
                for category in categories
            ]
            series.append((variant, values, colors[variant]))
        name = "behavior_success.svg"
        write_bar_chart(out / name, "Behavioral replay success by category", categories, series, percent=True)
        plots.append(name)

    runtime = report.get("runtime") or {}
    variants = [name for name in ("reference", "quant") if isinstance(runtime.get(name), dict)]
    footprint_keys = [
        ("Artifact GiB", "artifact_bytes"),
        ("Peak RSS GiB", "peak_rss_bytes"),
    ]
    footprint_keys = [
        (label, key)
        for label, key in footprint_keys
        if variants and all(finite(runtime[v].get(key)) is not None for v in variants)
    ]
    if footprint_keys:
        colors = {"reference": "#58a6ff", "quant": "#d2a8ff"}
        series = [
            (
                variant,
                [float(runtime[variant][key]) / 2**30 for _, key in footprint_keys],
                colors[variant],
            )
            for variant in variants
        ]
        name = "runtime_footprint.svg"
        write_bar_chart(
            out / name,
            "Artifact and serving-memory footprint",
            [label for label, _ in footprint_keys],
            series,
        )
        plots.append(name)
    perf_keys = [("Prefill tok/s", "prefill_tokens_per_second"), ("Decode tok/s", "decode_tokens_per_second")]
    perf_variants = [
        variant
        for variant in variants
        if any(finite(runtime[variant].get(key)) is not None for _, key in perf_keys)
    ]
    if perf_variants:
        series = []
        colors = {"reference": "#58a6ff", "quant": "#d2a8ff"}
        for variant in perf_variants:
            series.append((variant, [finite(runtime[variant].get(key)) or 0.0 for _, key in perf_keys], colors[variant]))
        name = "runtime_throughput.svg"
        write_bar_chart(out / name, "Runtime throughput", [label for label, _ in perf_keys], series)
        plots.append(name)

    tensor = report.get("tensor_error") or {}
    classes = tensor.get("classes", {}) if isinstance(tensor, dict) else {}
    if classes:
        labels = list(classes)
        weighted = [
            finite(classes[label].get("mean_weighted_nrmse"))
            if finite(classes[label].get("mean_weighted_nrmse")) is not None
            else (finite(classes[label].get("mean_nrmse")) or 0.0)
            for label in labels
        ]
        name = "tensor_reconstruction_error.svg"
        write_bar_chart(out / name, "Sampled reconstruction error by tensor class", labels, [("weighted NRMSE", weighted, "#ffa657")])
        plots.append(name)

    variants = report.get("variants") or []
    for metric, label, filename in (
        ("mean_kld", "Mean KL divergence (lower is better)", "size_vs_kld.svg"),
        ("p99_9_kld", "99.9% KL divergence (lower is better)", "size_vs_p99_9_kld.svg"),
        (
            "same_top_probability_pct",
            "Same top token (%) (higher is better)",
            "size_vs_same_top.svg",
        ),
        (
            "behavioral_pass_rate",
            "Behavioral pass rate (higher is better)",
            "size_vs_behavior.svg",
        ),
    ):
        points = [
            (str(row["name"]), float(row["size_gib"]), float(row[metric]))
            for row in variants
            if finite(row.get("size_gib")) is not None and finite(row.get(metric)) is not None
        ]
        if len(points) >= 2:
            write_scatter_chart(
                out / filename,
                f"Size / quality frontier: {label}",
                points,
                "Artifact size (GiB)",
                label,
            )
            plots.append(filename)

    modes = report.get("operating_modes") or []
    if modes:
        labels = [str(row["name"]) for row in modes]
        pass_rate = [finite(row.get("behavioral_pass_rate")) or 0.0 for row in modes]
        tool_result = [finite(row.get("tool_result_pass_rate")) or 0.0 for row in modes]
        identifier = [finite(row.get("identifier_integrity_rate")) or 0.0 for row in modes]
        name = "operating_mode_safety.svg"
        write_bar_chart(
            out / name,
            "Serving-mode behavioral safety",
            labels,
            [
                ("case pass", pass_rate, "#58a6ff"),
                ("tool-result pass", tool_result, "#3fb950"),
                ("identifier integrity", identifier, "#ffa657"),
            ],
            percent=True,
        )
        plots.append(name)
    return plots


def degradation_text(report: dict[str, Any]) -> str:
    scores = report.get("scores") or {}
    comparison = scores.get("comparison") or {}
    delta = finite(comparison.get("avg_nll_delta_quant_minus_reference"))
    if delta is None:
        distribution = report.get("distribution") or {}
        mean_kld = finite(distribution.get("mean_kld"))
        reference_ppl = finite(distribution.get("reference_perplexity"))
        quant_ppl = finite(distribution.get("quant_perplexity"))
        if mean_kld is None:
            behavior = report.get("behavior") or {}
            reference_rate = finite((behavior.get("reference") or {}).get("success_rate"))
            quant_rate = finite((behavior.get("quant") or {}).get("success_rate"))
            if reference_rate is not None and quant_rate is not None:
                return (
                    "Matched behavioral replay passed "
                    f"{quant_rate * 100:.1f}% for the candidate versus "
                    f"{reference_rate * 100:.1f}% for the upstream endpoint "
                    f"({(quant_rate - reference_rate) * 100:+.1f} percentage points). "
                    "This is an end-to-end result that includes source transformation, "
                    "engine, and serving-mode effects; matched-engine quantization-only "
                    "output degradation was not measured."
                )
            return "Quantization-only output degradation was not measured."
        details = [f"mean full-vocabulary KL divergence was {mean_kld:.6f}"]
        p99_9 = finite(distribution.get("p99_9_kld"))
        if p99_9 is not None:
            details.append(f"the 99.9th-percentile KL divergence was {p99_9:.6f}")
        if reference_ppl is not None and quant_ppl is not None and reference_ppl > 0:
            details.append(
                f"perplexity changed from {reference_ppl:.4f} to {quant_ppl:.4f} "
                f"({(quant_ppl / reference_ppl - 1) * 100:+.1f}%)"
            )
        return (
            "On the held-out logit corpus, "
            + ", and ".join(details)
            + ". These measurements describe the selected corpus, not a universal task-accuracy guarantee."
        )
    ratio = math.exp(delta)
    if delta <= 0:
        degree = "no aggregate loss was detected"
    elif delta <= 0.005:
        degree = "a very small aggregate shift was detected"
    elif delta <= 0.02:
        degree = "a small aggregate shift was detected"
    elif delta <= 0.05:
        degree = "a noticeable aggregate shift was detected"
    elif delta <= 0.15:
        degree = "a material aggregate shift was detected"
    else:
        degree = "a large aggregate shift was detected"
    return (
        f"On the frozen continuation set, {degree}: target-token NLL changed by "
        f"{delta:+.4f}, equivalent to a {((ratio - 1) * 100):+.1f}% perplexity ratio. "
        "This describes the measured corpus, not a universal task-accuracy guarantee."
    )


def write_markdown(path: Path, report: dict[str, Any], plots: list[str]) -> None:
    identity = report.get("identity", {})
    title = identity.get("title") or "Quantization quality report"
    lines = [f"# {md(title)}", "", f"Generated: `{report['provenance']['generated_at']}`", ""]
    gate_status = report.get("gates", {}).get("status", "not_gated")
    lines += [f"Release gate: **{gate_status.upper().replace('_', ' ')}**", ""]

    artifact_rows = []
    for role in ("upstream", "reference", "quant"):
        value = identity.get(role)
        if isinstance(value, dict):
            artifact_rows.append(
                (
                    role,
                    value.get("name") or "not recorded",
                    value.get("artifact") or value.get("fixture") or "not recorded",
                    value.get("sha256") or "not recorded",
                )
            )
    if artifact_rows:
        lines += ["## Artifacts", "", "| Role | Identity | Artifact or fixture | SHA-256 |", "|---|---|---|---|"]
        for role, name, artifact, digest in artifact_rows:
            lines.append(f"| {md(role)} | {md(name)} | {md(artifact)} | {md(digest)} |")
        quant = identity.get("quant")
        if isinstance(quant, dict) and quant.get("importance_matrix"):
            imatrix_line = f"Importance matrix: `{md(quant['importance_matrix'])}`"
            if quant.get("importance_matrix_sha256"):
                imatrix_line += f" (SHA-256 `{md(quant['importance_matrix_sha256'])}`)"
            lines += ["", imatrix_line]
        lines.append("")

    evaluation = identity.get("evaluation")
    if isinstance(evaluation, dict) and evaluation:
        lines += ["## Evaluation contract", ""]
        for key, value in evaluation.items():
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
            lines.append(f"- {md(key.replace('_', ' ').title())}: {md(rendered)}")
        lines.append("")

    lines += ["## Expected degradation", "", degradation_text(report), ""]

    scores = report.get("scores")
    if scores:
        ra, qa, comp = scores["reference"], scores["quant"], scores["comparison"]
        lines += [
            "The source column isolates transformations that happened before quantization; the quant-minus-source delta is the quantization-only estimate.",
            "",
            "| Metric | Source | Quant | Quant − source |",
            "|---|---:|---:|---:|",
            f"| Target-token NLL | {fmt(ra['avg_nll'])} | {fmt(qa['avg_nll'])} | {fmt(comp['avg_nll_delta_quant_minus_reference'], 4)} |",
            f"| Continuation perplexity | {fmt(ra['perplexity'])} | {fmt(qa['perplexity'])} | {pct((comp['perplexity_ratio_quant_over_reference'] or 1) - 1, signed=True)} |",
            f"| First token matches upstream | {pct(ra['official_first_token_match_rate'])} | {pct(qa['official_first_token_match_rate'])} | {pct(comp['official_first_token_match_rate_delta'], signed=True)} |",
            f"| Greedy common-prefix tokens | {fmt(ra['avg_greedy_lcp'], 2)} | {fmt(qa['avg_greedy_lcp'], 2)} | {fmt(comp['avg_greedy_lcp_delta'], 2)} |",
            f"| API top-N recall | {pct(ra.get('api_topn_recall'))} | {pct(qa.get('api_topn_recall'))} | — |",
            f"| API pair-order agreement | {pct(ra.get('api_pair_rate'))} | {pct(qa.get('api_pair_rate'))} | — |",
            "",
            f"Coverage: {comp['common_cases']} cases and {qa['target_tokens']} target tokens. Case wins (lower NLL): quant {comp['case_wins']['quant']}, source {comp['case_wins']['reference']}, ties {comp['case_wins']['ties']}.",
            "",
        ]

    distribution = report.get("distribution")
    if distribution:
        lines += ["## Full-logit distribution", "", "| Metric | Value |", "|---|---:|"]
        reference_ppl = finite(distribution.get("reference_perplexity"))
        quant_ppl = finite(distribution.get("quant_perplexity"))
        if reference_ppl is not None:
            lines.append(f"| Source perplexity | {reference_ppl:.6f} |")
        if quant_ppl is not None:
            lines.append(f"| Quant perplexity | {quant_ppl:.6f} |")
        if reference_ppl is not None and quant_ppl is not None and reference_ppl > 0:
            lines.append(
                f"| Quant/source perplexity change | {(quant_ppl / reference_ppl - 1) * 100:+.3f}% |"
            )
        for key, label, is_pct in (
            ("mean_kld", "Mean KL divergence", False),
            ("p99_9_kld", "99.9% KL divergence", False),
            ("maximum_kld", "Maximum KL divergence", False),
            ("same_top_probability_pct", "Same top token", True),
            ("mean_delta_probability_pct", "Mean correct-token probability change", True),
            ("rms_delta_probability_pct", "RMS probability change", True),
        ):
            value = distribution.get(key)
            if value is not None:
                rendered = f"{float(value):.3f}%" if is_pct else fmt(value, 6)
                lines.append(f"| {label} | {rendered} |")
        lines.append("")

    behavior = report.get("behavior")
    if behavior:
        lines += [
            "## Behavioral and agentic replay",
            "",
            "| Variant | Cases | Pass | Valid tool calls | Repetition | Raw DSML | Thinking tags | Identifier integrity |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for variant in ("reference", "quant"):
            if variant not in behavior:
                continue
            value = behavior[variant]
            lines.append(
                f"| {variant} | {value['cases']} | {pct(value.get('success_rate'))} | "
                f"{pct(value.get('tool_call_valid_rate'))} | {pct(value.get('repetition_detected_rate'))} | "
                f"{pct(value.get('dsml_leak_detected_rate'))} | {pct(value.get('thinking_leak_detected_rate'))} | "
                f"{pct(value.get('identifier_integrity_rate'))} |"
            )
        lines.append("")

    tensor_error = report.get("tensor_error")
    if isinstance(tensor_error, dict) and tensor_error.get("classes"):
        lines += [
            "## Sampled tensor reconstruction",
            "",
            "| Tensor class | Tensors | Mean NRMSE | Importance-weighted NRMSE | Mean cosine |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, value in tensor_error["classes"].items():
            lines.append(
                f"| {md(name)} | {value['tensors']} | {fmt(value.get('mean_nrmse'), 6)} | "
                f"{fmt(value.get('mean_weighted_nrmse'), 6)} | {fmt(value.get('mean_cosine'), 6)} |"
            )
        lines.append("")

    structural = report.get("structural_audit")
    if isinstance(structural, dict):
        lines += [
            "## GGUF structural audit",
            "",
            "| Complete | Version | Tensors | Actual bytes | Expected bytes | Trailing bytes |",
            "|---|---:|---:|---:|---:|---:|",
            f"| {'yes' if structural.get('complete') else 'no'} | "
            f"{fmt(structural.get('version'), 0)} | {fmt(structural.get('tensors'), 0)} | "
            f"{fmt(structural.get('actual_file_bytes'), 0)} | "
            f"{fmt(structural.get('expected_file_bytes'), 0)} | "
            f"{fmt(structural.get('trailing_bytes'), 0)} |",
            "",
        ]
        type_counts = structural.get("type_counts")
        if isinstance(type_counts, dict):
            lines += ["| Tensor type | Count |", "|---|---:|"]
            for tensor_type, count in sorted(type_counts.items()):
                lines.append(f"| {md(tensor_type)} | {fmt(count, 0)} |")
            lines.append("")

    runtime = report.get("runtime")
    if runtime:
        lines += ["## Size and runtime", "", "| Variant | Artifact GiB | Peak RSS GiB | Prefill tok/s | Decode tok/s |", "|---|---:|---:|---:|---:|"]
        for variant in ("reference", "quant"):
            if not isinstance(runtime.get(variant), dict):
                continue
            value = runtime[variant]
            artifact_bytes = finite(value.get("artifact_bytes"))
            peak_rss_bytes = finite(value.get("peak_rss_bytes"))
            artifact_text = (
                f"{artifact_bytes / 2**30:.2f}" if artifact_bytes is not None else "not measured"
            )
            rss_text = (
                f"{peak_rss_bytes / 2**30:.2f}" if peak_rss_bytes is not None else "not measured"
            )
            lines.append(f"| {variant} | {artifact_text} | {rss_text} | {fmt(value.get('prefill_tokens_per_second'), 2)} | {fmt(value.get('decode_tokens_per_second'), 2)} |")
        lines.append("")

    variants = report.get("variants")
    if variants:
        lines += [
            "## Variant quality frontier",
            "",
            "| Variant | Size GiB | BPW | PPL | Mean KLD | 99.9% KLD | Max KLD | Same top token | Behavior pass |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for variant in variants:
            same_top = variant.get("same_top_probability_pct")
            behavior_pass = variant.get("behavioral_pass_rate")
            lines.append(
                f"| {md(variant['name'])} | {fmt(variant.get('size_gib'), 2)} | "
                f"{fmt(variant.get('bpw'), 3)} | {fmt(variant.get('perplexity'), 4)} | "
                f"{fmt(variant.get('mean_kld'), 6)} | {fmt(variant.get('p99_9_kld'), 6)} | "
                f"{fmt(variant.get('maximum_kld'), 6)} | "
                f"{(fmt(same_top, 2) + '%') if same_top is not None else 'not measured'} | "
                f"{pct(behavior_pass) if behavior_pass is not None else 'not measured'} |"
            )
        lines.append("")

    modes = report.get("operating_modes")
    if modes:
        lines += [
            "## Serving-mode comparison",
            "",
            "| Mode | Artifact | Prefill | Drafter | Tool-result guard | Cases | Pass | Tool-result pass | Repetition | Identifier integrity | Prefill tok/s | Decode tok/s |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for mode in modes:
            lines.append(
                f"| {md(mode['name'])} | {md(mode.get('artifact', 'not recorded'))} | "
                f"{md(mode.get('prefill', 'not recorded'))} | {md(mode.get('drafter', 'not recorded'))} | "
                f"{md(mode.get('tool_result_guard', 'not recorded'))} | {fmt(mode.get('cases'), 0)} | "
                f"{pct(mode.get('behavioral_pass_rate'))} | {pct(mode.get('tool_result_pass_rate'))} | "
                f"{pct(mode.get('repetition_detected_rate'))} | "
                f"{pct(mode.get('identifier_integrity_rate'))} | "
                f"{fmt(mode.get('prefill_tokens_per_second'), 2)} | "
                f"{fmt(mode.get('decode_tokens_per_second'), 2)} |"
            )
        lines.append("")

    if report.get("gates", {}).get("results"):
        lines += ["## Release gates", "", "| Gate | Observed | Requirement | Status |", "|---|---:|---:|---|"]
        for gate in report["gates"]["results"]:
            lines.append(f"| {md(gate['label'])} | {fmt(gate['actual'], 6)} | {gate['op']} {fmt(gate['limit'], 6)} | {gate['status']} |")
        lines.append("")

    if plots:
        lines += ["## Plots", ""]
        for plot in plots:
            lines += [f"![{md(Path(plot).stem.replace('_', ' '))}]({plot})", ""]

    limitations = report.get("limitations", [])
    lines += ["## Interpretation and limitations", ""]
    for item in limitations:
        lines.append(f"- {md(item)}")
    lines += [
        "",
        "Perplexity and NLL are most meaningful between artifacts using the same tokenizer, prompt template, engine semantics, and corpus. Behavioral evaluations remain necessary because a small average logit shift can still flip a tool call or identifier.",
        "",
        "## Provenance", "",
        "```json", json.dumps(report["provenance"], indent=2, sort_keys=True), "```", "",
    ]
    notes = identity.get("notes") or []
    if notes:
        lines += ["## Release notes", ""] + [f"- {md(note)}" for note in notes] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="report identity, provenance, and gates JSON")
    parser.add_argument("--reference-scores", type=Path, help="quantization-source Dwarfstar score_official TSV")
    parser.add_argument("--quant-scores", type=Path, help="quant Dwarfstar score_official TSV")
    parser.add_argument("--distribution", type=Path, help="llama-perplexity log or normalized JSON")
    parser.add_argument("--behavior", type=Path, help="behavioral replay JSONL")
    parser.add_argument("--runtime", type=Path, help="runtime comparison JSON")
    parser.add_argument("--structural-audit", type=Path, help="GGUF structural audit JSON")
    parser.add_argument("--tensor-error", type=Path, help="sampled tensor error TSV")
    parser.add_argument(
        "--variants",
        type=Path,
        help="optional JSON/TSV multi-quant leaderboard for size/quality frontier plots",
    )
    parser.add_argument(
        "--modes",
        type=Path,
        help="optional JSON/TSV serving-mode comparison for safety and throughput plots",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if bool(args.reference_scores) != bool(args.quant_scores):
        die("--reference-scores and --quant-scores must be supplied together")
    for path in (
        args.metadata,
        args.reference_scores,
        args.quant_scores,
        args.distribution,
        args.behavior,
        args.runtime,
        args.structural_audit,
        args.tensor_error,
        args.variants,
        args.modes,
    ):
        if path is not None and not path.is_file():
            die(f"input does not exist: {path}")

    metadata = load_json(args.metadata)
    if not isinstance(metadata, dict):
        die("metadata must be a JSON object")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    inputs = {"metadata": input_provenance(args.metadata)}
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identity": metadata,
        "provenance": {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "generator": str(Path(__file__).resolve()),
            "generator_git_revision": git_revision(),
            "python": platform.python_version(),
            "host": platform.node(),
            "inputs": inputs,
        },
    }
    cases: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    if args.reference_scores and args.quant_scores:
        reference_rows, quant_rows = load_tsv(args.reference_scores), load_tsv(args.quant_scores)
        report["scores"], cases = compare_scores(reference_rows, quant_rows)
        inputs["reference_scores"] = input_provenance(args.reference_scores)
        inputs["quant_scores"] = input_provenance(args.quant_scores)
        with (args.out_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(cases[0]))
            writer.writeheader()
            writer.writerows(cases)
    if args.distribution:
        report["distribution"] = parse_distribution(args.distribution)
        inputs["distribution"] = input_provenance(args.distribution)
    if args.behavior:
        behavior_rows = load_jsonl(args.behavior)
        report["behavior"] = behavior_aggregate(behavior_rows)
        report["behavior_cases"] = behavior_rows
        write_rows_csv(args.out_dir / "behavior_cases.csv", behavior_rows)
        inputs["behavior"] = input_provenance(args.behavior)
    if args.runtime:
        runtime = load_json(args.runtime)
        if not isinstance(runtime, dict):
            die("runtime JSON must be an object")
        report["runtime"] = runtime
        inputs["runtime"] = input_provenance(args.runtime)
    if args.structural_audit:
        structural_audit = load_json(args.structural_audit)
        if not isinstance(structural_audit, dict):
            die("structural audit JSON must be an object")
        report["structural_audit"] = structural_audit
        inputs["structural_audit"] = input_provenance(args.structural_audit)
    if args.tensor_error:
        report["tensor_error"] = tensor_aggregate(load_tsv(args.tensor_error))
        inputs["tensor_error"] = input_provenance(args.tensor_error)
    if args.variants:
        report["variants"] = load_variants(args.variants)
        inputs["variants"] = input_provenance(args.variants)
    if args.modes:
        report["operating_modes"] = load_operating_modes(args.modes)
        inputs["operating_modes"] = input_provenance(args.modes)

    limitations = list(metadata.get("limitations") or [])
    if "distribution" not in report:
        limitations.append("Full-vocabulary KL divergence and same-top-token probability were not supplied.")
    if "behavior" not in report:
        limitations.append("Agent/tool behavioral replay was not supplied.")
    if "tensor_error" not in report:
        limitations.append("Sampled weight-reconstruction error was not supplied.")
    if "runtime" not in report:
        limitations.append("Runtime, memory, and throughput measurements were not supplied.")
    report["limitations"] = limitations
    report["gates"] = evaluate_gates(report, metadata.get("gates") or [])

    plots = make_plots(args.out_dir, report, cases)
    report["artifacts"] = {
        "markdown": "report.md",
        "json": "report.json",
        "case_metrics": "case_metrics.csv" if cases else None,
        "behavior_cases": "behavior_cases.csv" if behavior_rows else None,
        "plots": plots,
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(args.out_dir / "report.md", report, plots)
    print(f"report: {args.out_dir / 'report.md'}")
    print(f"machine data: {args.out_dir / 'report.json'}")
    print(f"release gate: {report['gates']['status']}")
    return 1 if report["gates"]["status"] in {"fail", "incomplete"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
