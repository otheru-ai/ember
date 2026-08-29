#!/usr/bin/env python3
"""Turn Qwen ROCMI4 dispatch telemetry into fail-closed evidence.

This parser intentionally treats a startup W4A8 marker as configuration only.
An enabled variant passes only after the real-weight differential run records
the expected exact-gfx1151 kernel launches and the MMVQ negative controls.
Logical q=4 uses Qwen's documented q=5 cached frontier.  Each control names the
exact target weight at both boundaries.  The parser permits the real MoE graph's
other dense/shared/routed events, but requires an ordered target-weight route,
target-weight IU4 kernel (or MMVQ negative), and target post-compute completion
inside that one boundary.  Events from another control cannot satisfy it.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SCHEMA = "ember.qwen3.8.kernel-runtime-evidence.v2"
ISA_ARCHIVE_SHA256 = (
    "82404f1126761b7877595b622afa7e1f311f2f41e89a3abe9aaf8ad045c082e2"
)
TAG = "[rocmi4-w4a8-dispatch]"
STARTUP = re.compile(
    r"ROCmI4 W4A8 IU4: exact experimental MMQ enabled for device (?P<device>[0-9]+); "
    r"activation_prepack=(?P<prepack>on|off)"
)
KERNEL = re.compile(
    r"\[rocmi4-w4a8-dispatch\] event=kernel "
    r"variant=(?P<variant>w4a8_iu4_(?:prepack|register_pack)) "
    r"op=(?P<op>dense|routed_expert) physical_q=(?P<q>[0-9]+) "
    r"type=Q4_0_ROCMI4 weight=(?P<weight>\S+) "
    r"device=(?P<device>[0-9]+) arch=gfx1151"
)
ROUTE = re.compile(
    r"\[rocmi4-w4a8-dispatch\] event=route "
    r"op=(?P<op>dense|routed_expert) physical_q=(?P<q>[0-9]+) "
    r"type=Q4_0_ROCMI4 path=(?P<path>mmvq|mmq|mmf|sync_fallback) "
    r"weight=(?P<weight>\S*) dst=(?P<dst>\S*)"
)
LOGICAL = re.compile(
    r"\[rocmi4-w4a8-dispatch\] event=logical_scope op=dense "
    r"logical_q=(?P<logical>[0-9]+) physical_q=(?P<physical>[0-9]+) "
    r"type=Q4_0_ROCMI4 execution=completed weight=(?P<weight>\S*)"
)
CONTROL = re.compile(
    r"\[rocmi4-w4a8-dispatch\] event=control "
    r"control_id=(?P<control_id>dense-q(?:1|4|5|16)|routed-expert-q(?:1|5|16)) "
    r"op=(?P<op>dense|routed_expert) logical_q=(?P<q>[0-9]+) "
    r"target_weight=(?P<weight>\S+) "
    r"phase=(?P<phase>begin|completed)"
)
POST_COMPUTE = re.compile(
    r"\[rocmi4-w4a8-dispatch\] event=post_compute "
    r"control_id=(?P<control_id>dense-q(?:1|4|5|16)|routed-expert-q(?:1|5|16)) "
    r"op=(?P<op>dense|routed_expert) logical_q=(?P<logical>[0-9]+) "
    r"physical_q=(?P<physical>[0-9]+) target_weight=(?P<weight>\S+) "
    r"execution=completed"
)
SUITE = re.compile(
    r"\[rocmi4-w4a8-dispatch\] event=control_suite "
    r"capability=(?P<capability>rocmi4_dense_and_routed|rocmi4_dense_only|"
    r"no_eligible_rocmi4_mmq) "
    r"dense_q=(?P<dense>1,4,5,16|none) "
    r"routed_expert_q=(?P<routed>1,5,16|none) execution=completed"
)


class EvidenceError(ValueError):
    pass


def bind_expected_capability(evidence: dict[str, object],
                             expected: str | None) -> dict[str, object]:
    if expected is None:
        return evidence
    actual = evidence.get("candidate_kernel_capability")
    if (actual is None and expected == "no_eligible_rocmi4_mmq" and
            evidence.get("dispatch_confirmation") ==
            "not_applicable_w4a8_not_configured"):
        enriched = dict(evidence)
        enriched.update({
            "candidate_kernel_capability": expected,
            "candidate_timing_kernel_mode":
                "not_applicable_no_eligible_rocmi4_mmq",
            "dispatch_confirmation":
                "not_applicable_no_eligible_rocmi4_mmq",
            "capability_confirmation":
                "quant_recipe_declares_no_eligible_rocmi4_mmq",
            "positive_controls": {}, "negative_controls": {},
            "ordered_control_ids": [], "observed_kernel_dispatches": [],
            "logical_dense_scopes": [],
        })
        return enriched
    if actual != expected:
        raise EvidenceError(
            f"runtime capability {actual!r} differs from expected {expected!r}")
    return evidence


def parse_log(text: str) -> dict[str, object]:
    marker_lines = [line[line.index("ROCmI4"):].strip()
                    for line in text.splitlines() if "ROCmI4 W4A" in line]
    known_startup = re.compile(
        r"ROCmI4 W4A8 IU4: exact experimental MMQ enabled for device [0-9]+; "
        r"activation_prepack=(?:on|off)|"
        r"ROCmI4 W4A4: enabled for device [0-9]+ "
        r"\(lossy prompt-processing path\)"
    )
    unknown = [line for line in marker_lines if known_startup.fullmatch(line) is None]
    if unknown:
        raise EvidenceError(f"unrecognized ROCmI4 mode markers: {unknown}")

    startups = list(STARTUP.finditer(text))
    states = {match.group("prepack") for match in startups}
    if len(states) > 1:
        raise EvidenceError(f"conflicting W4A8 variants: {sorted(states)}")
    w4a4 = "ROCmI4 W4A4:" in text
    if states and w4a4:
        raise EvidenceError("mutually exclusive W4A4 and W4A8 markers")

    tagged = [line.strip() for line in text.splitlines() if TAG in line]
    events: list[tuple[str, dict[str, str]]] = []
    for line in tagged:
        fragment = line[line.index(TAG):]
        for name, regex in (("kernel", KERNEL), ("route", ROUTE),
                            ("logical", LOGICAL), ("control", CONTROL),
                            ("post_compute", POST_COMPUTE), ("suite", SUITE)):
            match = regex.fullmatch(fragment)
            if match is not None:
                events.append((name, match.groupdict()))
                break
        else:
            raise EvidenceError(f"malformed dispatch marker: {fragment}")

    if not states and events:
        if (len(events) == 1 and events[0][0] == "suite"
                and events[0][1] == {
                    "capability": "no_eligible_rocmi4_mmq",
                    "dense": "none", "routed": "none",
                }):
            return {
                "schema": SCHEMA,
                "configured_mmq_mode": "exact_int8_mmq_control",
                "candidate_kernel_capability": "no_eligible_rocmi4_mmq",
                "candidate_timing_kernel_mode":
                    "not_applicable_no_eligible_rocmi4_mmq",
                "dispatch_confirmation":
                    "not_applicable_no_eligible_rocmi4_mmq",
                "w4a8_iu4_runtime_enabled": False,
                "activation_prepack": None, "passed": True,
                "positive_controls": {}, "negative_controls": {},
                "ordered_control_ids": [],
                "observed_kernel_dispatches": [],
                "logical_dense_scopes": [],
                "source": "differential-dispatch-server.log",
            }
        raise EvidenceError("dispatch controls ran without a configured W4A8 variant")
    if not states:
        return {
            "schema": SCHEMA,
            "configured_mmq_mode": ("lossy_w4a4_mmq" if w4a4
                                    else "exact_int8_mmq_control"),
            "dispatch_confirmation": "not_applicable_w4a8_not_configured",
            "w4a8_iu4_runtime_enabled": False,
            "activation_prepack": None,
            "passed": True,
            "source": "differential-dispatch-server.log",
        }

    state = next(iter(states))
    expected_variant = ("w4a8_iu4_prepack" if state == "on"
                        else "w4a8_iu4_register_pack")
    suite_indices = [index for index, event in enumerate(events)
                     if event[0] == "suite"]
    if len(suite_indices) != 1:
        raise EvidenceError(
            "expected one completed real-weight control suite, observed "
            f"{len(suite_indices)}")

    suite_index = suite_indices[0]
    suite = events[suite_index][1]
    capability = suite["capability"]
    suite_contract = {
        "rocmi4_dense_and_routed": ("1,4,5,16", "1,5,16"),
        "rocmi4_dense_only": ("1,4,5,16", "none"),
        "no_eligible_rocmi4_mmq": ("none", "none"),
    }
    if (suite["dense"], suite["routed"]) != suite_contract[capability]:
        raise EvidenceError("control-suite capability and declared controls differ")
    if (capability == "no_eligible_rocmi4_mmq"
            and any(kind != "suite" for kind, _row in events)):
        raise EvidenceError(
            "not-applicable candidate emitted ROCMI4 dispatch control events")

    dense_controls = (
        ("dense-q1", "dense", 1, 1, "mmvq", False),
        ("dense-q4", "dense", 4, 5, "mmq", True),
        ("dense-q5", "dense", 5, 5, "mmq", True),
        ("dense-q16", "dense", 16, 16, "mmq", True),
    )
    routed_controls = (
        ("routed-expert-q1", "routed_expert", 1, 1, "mmvq", False),
        ("routed-expert-q5", "routed_expert", 5, 5, "mmvq", False),
        ("routed-expert-q16", "routed_expert", 16, 16, "mmq", True),
    )
    expected_controls = (() if capability == "no_eligible_rocmi4_mmq"
                         else dense_controls + (routed_controls if capability ==
                                                "rocmi4_dense_and_routed" else ()))
    control_boundaries = [
        (index, row) for index, (kind, row) in enumerate(events)
        if kind == "control"
    ]
    if len(control_boundaries) != len(expected_controls) * 2:
        raise EvidenceError(
            "ordered per-control dispatch evidence count differs from capability")
    validated: list[tuple[str, dict[str, str]]] = []
    cursor = 0
    prior_end = -1
    for control_id, op, logical_q, physical_q, path, needs_kernel in expected_controls:
        begin_index, begin = control_boundaries[cursor]
        end_index, end = control_boundaries[cursor + 1]
        cursor += 2
        expected_boundary = {"control_id": control_id, "op": op,
                             "q": str(logical_q)}
        if (begin_index <= prior_end or begin_index >= end_index or end_index >= suite_index
                or begin.get("phase") != "begin" or end.get("phase") != "completed"
                or any(begin.get(key) != value for key, value in expected_boundary.items())
                or any(end.get(key) != value for key, value in expected_boundary.items())
                or not begin.get("weight") or begin.get("weight") != end.get("weight")):
            raise EvidenceError(
                f"ordered dispatch evidence mismatch for control {control_id}")
        target_weight = begin["weight"]
        scope = events[begin_index + 1:end_index]
        target_routes = [(index, row) for index, (kind, row) in enumerate(scope)
                         if kind == "route" and row.get("weight") == target_weight]
        if len(target_routes) != 1:
            raise EvidenceError(
                f"control {control_id} did not route its exact target weight once")
        route_index, route = target_routes[0]
        if (route.get("op") != op or route.get("q") != str(physical_q)
                or route.get("path") != path):
            raise EvidenceError(
                f"ordered dispatch evidence mismatch for target route {control_id}")
        target_kernels = [(index, row) for index, (kind, row) in enumerate(scope)
                          if kind == "kernel" and row.get("weight") == target_weight]
        if needs_kernel:
            if (len(target_kernels) != 1 or target_kernels[0][0] <= route_index
                    or target_kernels[0][1].get("op") != op
                    or target_kernels[0][1].get("q") != str(physical_q)
                    or target_kernels[0][1].get("variant") != expected_variant):
                raise EvidenceError(
                    f"ordered dispatch evidence mismatch for target kernel {control_id}")
            compute_predecessor = target_kernels[0][0]
        else:
            if target_kernels:
                raise EvidenceError(
                    f"MMVQ negative control {control_id} launched a W4A8 kernel")
            compute_predecessor = route_index
        completions = [(index, row) for index, (kind, row) in enumerate(scope)
                       if kind == "post_compute" and row.get("control_id") == control_id]
        if (len(completions) != 1 or completions[0][0] <= compute_predecessor
                or completions[0][1] != {
                    "control_id": control_id, "op": op,
                    "logical": str(logical_q), "physical": str(physical_q),
                    "weight": target_weight,
                }):
            raise EvidenceError(
                f"control {control_id} lacks target-bound post-compute completion")
        validated.extend(scope)
        prior_end = end_index
    if prior_end >= suite_index:
        raise EvidenceError("control suite marker precedes a completed control")
    if any(kind in {"control", "post_compute"}
           for kind, _row in events[suite_index + 1:]):
        raise EvidenceError("control evidence continues after the suite marker")

    kernels = [row for kind, row in validated if kind == "kernel"]
    routes = [row for kind, row in validated if kind == "route"]
    logical = [row for kind, row in validated if kind == "logical"]
    variants = {row["variant"] for row in kernels}
    if capability != "no_eligible_rocmi4_mmq" and variants != {expected_variant}:
        raise EvidenceError(
            f"actual kernel variants {sorted(variants)} do not match startup "
            f"variant {expected_variant}")
    startup_devices = {match.group("device") for match in startups}
    kernel_devices = {row["device"] for row in kernels}
    if (capability != "no_eligible_rocmi4_mmq"
            and (not kernel_devices or not kernel_devices.issubset(startup_devices))):
        raise EvidenceError("kernel dispatch device is not startup-bound")

    kernel_set = {(row["op"], int(row["q"])) for row in kernels}
    route_set = {(row["op"], int(row["q"]), row["path"])
                 for row in routes}
    logical_set = {(int(row["logical"]), int(row["physical"]))
                   for row in logical}

    # Exact ordered validation above establishes each of these facts inside
    # the matching begin/completed control boundary. They are named separately
    # in retained evidence for downstream policy checks.
    positive = {control_id: True for control_id, _op, _logical, _physical,
                _path, needs_kernel in expected_controls if needs_kernel}
    negative = {control_id: True for control_id, _op, _logical, _physical,
                _path, needs_kernel in expected_controls if not needs_kernel}

    return {
        "schema": SCHEMA,
        "configured_mmq_mode": expected_variant,
        "candidate_kernel_capability": capability,
        "candidate_timing_kernel_mode": (
            expected_variant if capability != "no_eligible_rocmi4_mmq"
            else "not_applicable_no_eligible_rocmi4_mmq"),
        "dispatch_confirmation": (
            "actual_real_weight_dense_and_routed_launches"
            if capability == "rocmi4_dense_and_routed" else
            "actual_real_weight_dense_launches"
            if capability == "rocmi4_dense_only" else
            "not_applicable_no_eligible_rocmi4_mmq"),
        "w4a8_iu4_runtime_enabled": True,
        "activation_prepack": state == "on",
        "passed": True,
        "positive_controls": positive,
        "negative_controls": negative,
        "ordered_control_ids": [item[0] for item in expected_controls],
        "observed_kernel_dispatches": [
            {"op": op, "physical_q": q}
            for op, q in sorted(kernel_set)
        ],
        "logical_dense_scopes": [
            {"logical_q": logical_q, "physical_q": physical_q}
            for logical_q, physical_q in sorted(logical_set)
        ],
        "isa_contract": {
            "architecture": "gfx1151",
            "instruction": "V_WMMA_I32_16X16X16_IU4",
            "opcode": 69,
            "machine_readable_isa_archive_sha256": ISA_ARCHIVE_SHA256,
            "qualification": "runtime_dispatch_plus_separate_saved_isa_gate",
        },
        "source": "differential-dispatch-server.log",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-capability",
        choices=("rocmi4_dense_and_routed", "rocmi4_dense_only",
                 "no_eligible_rocmi4_mmq"),
        help="capability already bound to the candidate quant recipe")
    args = parser.parse_args()
    try:
        evidence = parse_log(args.log.read_text(encoding="utf-8", errors="replace"))
        evidence = bind_expected_capability(evidence, args.expected_capability)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(evidence, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except (OSError, EvidenceError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
