#!/usr/bin/env python3
"""GPU-free contract tests for container and operator scripts."""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "docker" / "entrypoint.sh"
COLLECT_RUNTIME = ROOT / "docker" / "collect-runtime.sh"
DOCKERFILE = ROOT / "docker" / "Dockerfile"
CONTAINER_WORKFLOW = ROOT / ".forgejo" / "workflows" / "container.yml"
GITHUB_CI = ROOT / ".github" / "workflows" / "ci.yml"
GITHUB_CONTAINER = ROOT / ".github" / "workflows" / "container.yml"
GITHUB_CERTIFY = ROOT / ".github" / "workflows" / "gfx1151-certify.yml"
SERVER_MAIN = ROOT / "src" / "server" / "main.c"
GITHUB_RELEASE_NOTES = ROOT / ".github" / "workflows" / "release-notes.yml"
FORGEJO_CI = ROOT / ".forgejo" / "workflows" / "ci.yml"
COMPOSE = ROOT / "compose.yaml"
COMPOSE_BUILD = ROOT / "compose.build.yaml"
COMPOSE_XDNA = ROOT / "compose.xdna.yaml"
PREFLIGHT = ROOT / "scripts" / "preflight.sh"
SMOKE = ROOT / "scripts" / "smoke_test.sh"


class ReleaseScriptTests(unittest.TestCase):
    @staticmethod
    def qwen_deployment_fixture(
        root: pathlib.Path,
    ) -> tuple[dict[str, str], dict[str, pathlib.Path]]:
        names = {
            "model": (
                "Qwen3.8-Flash-Next-Heretic-ROCmI4-Strix-Halo-"
                "00001-of-00002.gguf"
            ),
            "model_2": (
                "Qwen3.8-Flash-Next-Heretic-ROCmI4-Strix-Halo-"
                "00002-of-00002.gguf"
            ),
            "mtp": "Qwen3.8-Flash-Next-MTP-ROCmI4-Strix-Halo.gguf",
            "mmproj": "Qwen3.8-Flash-Next-BF16-mmproj.gguf",
            "vision_vocab": "Qwen3.8-Flash-Next-vocab-only.gguf",
        }
        paths = {key: root / name for key, name in names.items()}
        for key, path in paths.items():
            path.write_bytes(f"release-test-{key}".encode())
        provider = root / "libember_qwen4exp_vision_provider.so"
        provider.write_bytes(b"release-test-provider")
        paths["provider"] = provider
        checksums = root / "SHA256SUMS"
        checksums.write_text("".join(
            f"{hashlib.sha256(paths[key].read_bytes()).hexdigest()}  {names[key]}\n"
            for key in ("model", "model_2", "mtp", "mmproj", "vision_vocab")
        ))
        paths["checksums"] = checksums
        env = {
            "EMBER_SKIP_DEVICE_CHECK": "1",
            "EMBER_DEPLOYMENT_MODE": "qwen3.8-flash-next",
            "EMBER_MODEL_DIR": str(root),
            "EMBER_QWEN_MODEL": str(paths["model"]),
            "EMBER_QWEN_SHA256SUMS": str(checksums),
            "EMBER_QWEN_SHA256SUMS_SHA256": hashlib.sha256(
                checksums.read_bytes()
            ).hexdigest(),
            "DFLASH_QWEN_MTP": str(paths["mtp"]),
            "DFLASH_QWEN_MTP_DEPTH": "3",
            "DFLASH_QWEN_VISION_MMPROJ": str(paths["mmproj"]),
            "DFLASH_QWEN_VISION_TEXT_MODEL": str(paths["vision_vocab"]),
            "DFLASH_QWEN_VISION_PROVIDER": str(provider),
            "EMBER_KV_CACHE_DIR": str(root),
            "EMBER_SEGVTRACE": "",
        }
        return env, paths

    @staticmethod
    def docker_stage(dockerfile: str, name: str) -> str:
        match = re.search(
            rf"^FROM\s+\S+\s+AS\s+{re.escape(name)}\s*$",
            dockerfile,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        if not match:
            raise AssertionError(f"Docker stage not found: {name}")
        following = re.search(r"^FROM\s+", dockerfile[match.end():], re.MULTILINE)
        end = match.end() + following.start() if following else len(dockerfile)
        return dockerfile[match.end():end]

    def test_default_model_is_immutably_pinned(self) -> None:
        script = ENTRYPOINT.read_text()
        self.assertIn("/resolve/$revision/$artifact_file", script)
        self.assertNotIn("/resolve/main/", script)
        self.assertIn(
            "a936e0a514385c8ae964c0f42263a4314a34fbc6efea9d9aced5320f320a3d54",
            script,
        )
        self.assertIn(
            "1a01c80eceae302bcc1d70836759ee97974d7983c5084ef43f6ef772a8970ae6",
            script,
        )
        self.assertIn("9fe32d8d4a1abed16c84e2636b26950232869929", script)
        self.assertNotIn("EMBER_MODEL:-", script)
        self.assertNotIn("EMBER_DRAFT_MODEL:-", script)
        self.assertNotIn("EMBER_MODEL_SHA256-", script)
        self.assertNotIn("EMBER_DRAFT_MODEL_SHA256-", script)

    def test_shell_syntax(self) -> None:
        for script in (
            ENTRYPOINT,
            COLLECT_RUNTIME,
            PREFLIGHT,
            SMOKE,
        ):
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_local_compose_builds_use_host_network(self) -> None:
        # The supported WSL host intentionally disables Docker's unusable
        # bridge/iptables path. Local builds must not implicitly select it.
        for compose in (COMPOSE_BUILD, COMPOSE_XDNA):
            self.assertIn("network: host", compose.read_text())

        compose = COMPOSE.read_text()
        ember_dev = compose[compose.index("  ember-dev:"):]
        self.assertIn("network: host", ember_dev)

    def test_container_targets_separate_toolchain_from_runtime(self) -> None:
        dockerfile = DOCKERFILE.read_text()
        self.assertIn("AS dev", dockerfile)
        self.assertIn("AS release", dockerfile)
        self.assertGreaterEqual(dockerfile.count("@sha256:"), 2)
        release = self.docker_stage(dockerfile, "release")
        self.assertNotIn("build-essential", release)
        self.assertNotIn("cmake --build", release)
        self.assertIn("COPY --from=dev /ember-runtime/ /", release)
        # GHCR links a package to its repository from this label; without it a
        # package can end up unlinked and GITHUB_TOKEN cannot push to it.
        self.assertIn(
            'org.opencontainers.image.source="https://github.com/otheru-ai/ember"',
            release,
        )
        self.assertIn("/usr/share/licenses/ember/", release)
        self.assertIn("blas_lib_gfx1151.kpack", dockerfile)
        self.assertIn("rocblas/library/gfx1151", dockerfile)
        self.assertIn("TensileLibrary_lazy_gfx1151.dat", dockerfile)


    def test_xdna_image_keeps_host_driver_outside_container(self) -> None:
        dockerfile = DOCKERFILE.read_text()
        compose = COMPOSE_XDNA.read_text()
        runtime = self.docker_stage(dockerfile, "release-xdna")
        self.assertIn("COPY --from=xdna-userspace-build", runtime)
        self.assertIn("libember_xdna_dspark.so", runtime)
        self.assertNotIn("amdxdna.ko", runtime)
        self.assertIn("SKIP_KMOD=ON", dockerfile)
        self.assertIn("/dev/accel/accel0:/dev/accel/accel0", compose)
        self.assertIn("memlock:", compose)
        self.assertIn("DFLASH_DSPARK_XDNA_PLUGIN", compose)
        self.assertIn("DFLASH_DSPARK_XDNA_GPU_MAIN", compose)
        self.assertIn('command: ["--batch-sessions"', compose)
        self.assertNotIn("DFLASH_MOE_XDNA_PLUGIN:", compose)

    def test_compose_pulls_release_and_keeps_source_build_explicit(self) -> None:
        compose = COMPOSE.read_text()
        build = COMPOSE_BUILD.read_text()
        release_service = compose.split("  ember-dev:", 1)[0]
        version = (ROOT / "VERSION").read_text().strip()
        self.assertIn(f"ghcr.io/otheru-ai/ember:{version}", release_service)
        self.assertIn("pull_policy: always", release_service)
        self.assertNotIn("build:", release_service)
        # List form, not mapping form: a bare NAME entry forwards the
        # operator's value only when they set one, which is what lets this file
        # stay authoritative without restating a default that already lives in
        # the binary. Assert the syntax the file actually uses so a silent
        # switch back to mapping form fails here.
        self.assertIn("- EMBER_HOST=${EMBER_HOST:-127.0.0.1}", release_service)
        self.assertIn(
            "- EMBER_VERIFY_EXISTING_SHA256=${EMBER_VERIFY_EXISTING_SHA256:-1}",
            release_service,
        )
        # Pass-through entries carry no "=" at all.
        for name in (
            "DFLASH_DS4_SPEC",
            "DFLASH_DS4_SPEC_MAX_CTX",
            "DFLASH_DS4_Q5_VERIFY",
            "DFLASH_DS4_SPEC_Q",
            "LUCE_MMVQ_MAX_NCOLS",
        ):
            self.assertIn(f"\n      - {name}\n", release_service)
        self.assertIn("$${EMBER_HOST:-127.0.0.1}", release_service)
        self.assertIn("target: release", build)
        self.assertIn("pull_policy: build", build)

    def test_compose_exposes_explicit_qwen_deployment_contract(self) -> None:
        release_service = COMPOSE.read_text().split("  ember-dev:", 1)[0]
        self.assertIn(
            "- EMBER_DEPLOYMENT_MODE=${EMBER_DEPLOYMENT_MODE:-deepseek-v4-flash}",
            release_service,
        )
        for name in (
            "EMBER_QWEN_MODEL",
            "EMBER_QWEN_SHA256SUMS",
            "EMBER_QWEN_SHA256SUMS_SHA256",
            "DFLASH_QWEN_MTP",
            "DFLASH_QWEN_MTP_DEPTH",
            "DFLASH_QWEN_VISION_MMPROJ",
            "DFLASH_QWEN_VISION_TEXT_MODEL",
        ):
            self.assertIn(f"\n      - {name}\n", release_service)

    def test_container_publish_is_sha_and_hardware_gated(self) -> None:
        workflow = CONTAINER_WORKFLOW.read_text()
        self.assertIn("needs: source-gate", workflow)
        self.assertGreaterEqual(workflow.count("git rev-parse HEAD"), 2)
        self.assertIn("EMBER_GFX1151_CERTIFIED_SHA", workflow)
        self.assertIn("ctest --test-dir build", workflow)
        self.assertIn('tag_args+=(--tag "$image:latest")', workflow)
        self.assertIn("aquasec/trivy:0.73.0@sha256:", workflow)
        self.assertIn("--severity CRITICAL", workflow)

    def test_github_workflows_split_hosted_build_and_halo_certification(self) -> None:
        ci = GITHUB_CI.read_text()
        container = GITHUB_CONTAINER.read_text()
        certify = GITHUB_CERTIFY.read_text()
        self.assertIn("runs-on: ubuntu-24.04", ci)
        self.assertNotIn("self-hosted", ci)
        self.assertIn("actionlint_1.7.12_linux_amd64.tar.gz", ci)
        self.assertIn("vars.EMBER_BUILD_RUNNER || 'ubuntu-latest-8-cores'", container)
        self.assertIn("timeout-minutes:", container)
        self.assertIn("packages: write", container)
        self.assertIn("ghcr.io/${GITHUB_REPOSITORY,,}", container)
        self.assertIn("EMBER_GFX1151_CERTIFIED_SHA", container)
        self.assertIn("runs-on: [self-hosted, linux, x64, gfx1151]", certify)
        self.assertIn("workflow_call:", certify)
        self.assertNotIn("environment: gfx1151-certification", certify)
        self.assertIn("vars.EMBER_CERT_MODEL_PATH", certify)
        self.assertIn("vars.EMBER_CERT_DRAFT_PATH", certify)
        self.assertIn("Quiesce production for exclusive GPU validation", certify)
        self.assertIn("Restore production", certify)
        self.assertIn("/usr/local/sbin/ember-cert-production stop", certify)
        self.assertIn("/usr/local/sbin/ember-cert-production start", certify)
        # One gfx1151 and 125 GiB: a second model-loading process does not
        # fail, it silently degrades to hybrid expert placement. Certification
        # must hold the documented lock, not merely quiesce production.
        self.assertIn("ember-cert-production mask", certify)
        self.assertIn("ember-cert-production unmask", certify)
        self.assertIn("production was restarted during certification", certify)
        # Through the wrapper: /root is dr-xr-x--- and the runner cannot open
        # the lock file, so a direct flock here fails with EACCES and the wait
        # that follows it spins instead of holding anything.
        self.assertIn("ember-gpu-lock acquire", certify)
        self.assertIn("ember-gpu-lock release", certify)
        self.assertIn("restore=service", certify)
        self.assertIn("iommu|amd_iommu", certify)
        self.assertIn("iflag=direct", certify)
        self.assertIn(".ember-model-integrity-v1.json", certify)
        self.assertIn("integrity cache hit", certify)
        self.assertIn("docker stop --timeout", certify)
        self.assertIn("DOCKER_CONFIG", certify)
        self.assertIn("MemAvailable", certify)
        self.assertIn("less than 100 GiB", certify)
        self.assertIn("--validate-gemm-batch 64", certify)
        self.assertIn("--validate-prompt", certify)
        self.assertIn("Validation sentence", certify)
        self.assertIn('report["disk"]["checked"]', certify)
        self.assertIn('report["spec"]["checked"]', certify)
        self.assertIn("--batch-sessions 1", certify)
        self.assertIn(
            "a936e0a514385c8ae964c0f42263a4314a34fbc6efea9d9aced5320f320a3d54",
            certify,
        )
        self.assertIn(
            "1a01c80eceae302bcc1d70836759ee97974d7983c5084ef43f6ef772a8970ae6",
            certify,
        )
        self.assertIn("org.opencontainers.image.revision", certify)
        # Certification validates the published image and nothing else: no
        # working tree is checked out onto the machine that holds production
        # and the GPU. Benchmarking lives in its own job after promotion,
        # where a checkout is fine and a failure cannot cost a release.
        certify_job = certify.split("\n  promote:", 1)[0]
        self.assertNotIn("actions/checkout", certify_job)
        self.assertIn('"$source_sha" "$source_output" <<\'PY\'', certify_job)
        self.assertIn("source_sha, source_output = sys.argv[1:]", certify_job)
        self.assertIn("GHCR_TOKEN: ${{ secrets.GITHUB_TOKEN }}", certify_job)
        self.assertIn('echo "$GHCR_TOKEN" | docker login ghcr.io', certify_job)
        self.assertIn("-from-([0-9a-f]{40})", certify_job)
        self.assertIn("PROFILE_RESUME_ARG", certify_job)
        self.assertIn("  qwen-resume-q3-timing:", certify_job)
        self.assertIn("--calibrate-qwen-shapes", certify_job)
        self.assertIn("No model", certify_job)
        self.assertIn("integrity hash or quant construction is repeated", certify_job)
        self.assertIn('test "$current_engine_sha" = "${retained[2]}"', certify_job)
        self.assertIn('len(row.get("counter_files") or []) != 2', certify_job)
        self.assertIn("--require-memory-gate", certify_job)
        self.assertNotIn("benchmark_bundle.sh", certify_job)
        self.assertIn("  benchmark:", certify)
        self.assertIn("needs: [certify, promote]", certify)
        self.assertIn("benchmark_bundle.sh", certify)
        self.assertIn("build_perf_site_data.py", certify)
        self.assertIn("Promote certified candidate", certify)
        self.assertIn("ci/release_changelog.py prepare", certify)
        self.assertIn("git push --atomic", certify)
        self.assertIn("FORGEJO_RELEASE_SSH_KEY", certify)
        self.assertIn("StrictHostKeyChecking=yes", certify)
        self.assertIn("RELEASE_AUTOMATION_TOKEN", certify)
        self.assertIn("expected=(CHANGELOG.md compose.yaml VERSION)", certify)
        triggers = certify.split("permissions:", 1)[0]
        self.assertNotIn("pull_request", triggers)

    def test_github_release_candidate_is_gated_and_automatic(self) -> None:
        ci = GITHUB_CI.read_text()
        container = GITHUB_CONTAINER.read_text()
        release_notes = GITHUB_RELEASE_NOTES.read_text()
        forgejo_container = CONTAINER_WORKFLOW.read_text()
        self.assertIn("publish-candidate:", ci)
        # The scope job gates the expensive pair, so a documentation-only push
        # does not take the gfx1151 box for two hours; both must depend on it.
        self.assertIn(
            "needs: [invariants, build-test, sanitizers, analyzer, coverage, scope]",
            ci,
        )
        self.assertIn("scope:", ci)
        self.assertIn("needs.scope.outputs.code == 'true'", ci)
        self.assertIn("uses: ./.github/workflows/container.yml", ci)
        self.assertIn("workflow_call:", container)
        self.assertIn('git rev-parse "$publish_sha^"', container)
        self.assertIn("expected=(CHANGELOG.md compose.yaml VERSION)", container)
        self.assertIn("release_tag:", container)
        self.assertIn("inputs.release_tag || github.sha", container)
        self.assertIn("workflow_run:", release_notes)
        self.assertIn("workflow_dispatch:", release_notes)
        self.assertIn("inputs.release_tag || github.event.workflow_run.head_branch", release_notes)
        self.assertIn("conclusion == 'success'", release_notes)
        self.assertIn("ci/release_changelog.py notes", release_notes)
        self.assertIn("gh release create", release_notes)
        self.assertIn("!startsWith(github.event.head_commit.message", ci)
        self.assertIn("certify-and-release:", ci)
        self.assertIn("needs: [publish-candidate, scope]", ci)
        self.assertIn("uses: ./.github/workflows/gfx1151-certify.yml", ci)
        self.assertIn("commit_sha: ${{ github.sha }}", ci)
        self.assertIn("secrets: inherit", ci)
        self.assertIn("actions/cache@caa296126883cff596d87d8935842f9db880ef25", ci)
        self.assertIn("actions/cache@caa296126883cff596d87d8935842f9db880ef25", container)
        self.assertIn("EMBER_BUILDX_BUILDER", container)
        self.assertIn("ember-trivy-cache", container)
        self.assertIn("advice.detachedHead", container)
        self.assertNotIn("tags: ['v*']", forgejo_container)

    def test_counter_calibration_does_not_reacquire_workflow_gpu_lock(self) -> None:
        certify = GITHUB_CERTIFY.read_text()
        calibration = certify.split(
            "\n  qwen-calibrate-counter-units:", 1
        )[1].split("\n  qwen-resume-q3-timing:", 1)[0]
        self.assertIn(
            '"$CALIBRATION_SOURCE/scripts/calibrate_counter_units.sh" \\\n'
            '            --no-quiesce --image "$CALIBRATION_IMAGE"',
            calibration,
        )
        self.assertIn(
            "/ember/share/benchmark/gfx1151-rocm10-counter-calibration.json",
            certify,
        )
        self.assertIn("--counter-calibration", certify)

    def test_qwen_control_conversion_is_exact_bounded_and_recoverable(self) -> None:
        container = GITHUB_CONTAINER.read_text()
        certify = GITHUB_CERTIFY.read_text()
        dockerfile = DOCKERFILE.read_text()
        self.assertIn('tag "$image:dev-sha-$short_sha"', container)
        self.assertIn("--target dev", container)
        self.assertIn("dev-image-metadata.json", container)
        dev_stage = dockerfile.split("FROM toolchain AS dev", 1)[1].split(
            "FROM ${RUNTIME_IMAGE} AS release", 1
        )[0]
        self.assertIn("ARG EMBER_VERSION", dev_stage)
        self.assertIn("ARG EMBER_VCS_REF", dev_stage)
        self.assertIn(
            'org.opencontainers.image.revision="${EMBER_VCS_REF}"', dev_stage
        )
        self.assertIn(
            'org.opencontainers.image.version="${EMBER_VERSION}"', dev_stage
        )
        self.assertIn("ember-gguf-quantize ember-token-dump", dockerfile)
        self.assertIn("python3-venv git time", dockerfile)
        self.assertIn("qwen-convert-control:", certify)
        self.assertIn("--bounded-memory-temp", certify)
        self.assertIn("--stock-control", certify)
        self.assertIn("--gguf-splitter", certify)
        self.assertIn("scripts/qwen_mtp_export.py", certify)
        self.assertIn("Qwen3.8-Flash-Next-MTP-ROCmI4-Strix-Halo.gguf", certify)
        self.assertIn("--network none", certify)
        self.assertIn("/usr/bin/time -v", certify)
        self.assertIn("final_release_eligible", certify)
        self.assertIn("Stock-Control", certify)
        self.assertIn("qwen-docker-$GITHUB_RUN_ID", certify)
        self.assertIn("Remove temporary registry credentials", certify)
        control = certify.split("\n  qwen-convert-control:", 1)[1]
        self.assertIn('-e LLAMA_REVISION="$LLAMA_REVISION"', control)
        self.assertIn('[[ "$LLAMA_REVISION" == "$commit"* ]]', control)
        self.assertLess(control.index('[[ "$LLAMA_REVISION" == "$commit"* ]]'),
                        control.index("ember-gpu-lock acquire"))
        for repository in (
            "/ember", "/qwen-work/tooling/llama.cpp",
            "/qwen-work/tooling/ROCmFPX",
        ):
            self.assertIn(f"safe.directory {repository}", control)
        self.assertNotIn("safe.directory '*'", control)
        for command in (
            "ember-gpu-lock acquire", "ember-gpu-lock release",
            "ember-cert-production stop", "ember-cert-production mask",
            "ember-cert-production unmask", "ember-cert-production start",
        ):
            self.assertIn(command, control)
        self.assertGreaterEqual(control.count("if: ${{ always()"), 4)

    def test_qwen_stock_capture_is_digest_bound_no_clobber_and_nonpublishing(self) -> None:
        certify = GITHUB_CERTIFY.read_text()
        self.assertIn("qwen-capture-control:", certify)
        capture = certify.split("\n  qwen-capture-control:", 1)[1]
        self.assertIn("startsWith(inputs.release_version, 'qwen-capture-control')", capture)
        self.assertIn(":dev-sha-${CAPTURE_TOOL_SHA:0:12}", capture)
        self.assertIn("QWEN_DEV_IMAGE_DIGEST", capture)
        self.assertIn("EMBER_CONFIGURED_GIT_HEAD:STRING", capture)
        self.assertIn("test \"$(git rev-parse HEAD)\" = \"$CAPTURE_TOOL_SHA\"", capture)
        self.assertIn("git merge-base --is-ancestor \"$TARGET_SHA\" \"$CAPTURE_TOOL_SHA\"", capture)
        self.assertIn("--tool-revision \"$CAPTURE_TOOL_SHA\"", capture)
        self.assertIn("--artifact-revision \"$TARGET_SHA\"", capture)
        self.assertIn('manifest["model"]["quantizer_ember_revision"]', capture)
        self.assertIn('manifest["image"]["ember_revision"]', capture)
        self.assertIn(
            "/srv/ember/qwen3.8-otheru-corpus-${OTHERU_REVISION:0:8}", capture,
        )
        recipe_digest = hashlib.sha256(
            (ROOT / "share" / "quant_eval" /
             "qwen3.8-flash-next-bakeoff.json").read_bytes()
        ).hexdigest()
        for digest in (
            "19c70ad1ce7664b58fbaa854f7a80bc50868873a89e44459002b634137d5cc1d",
            "a41997529ad28af7234e036f05bd9bca39c504f8ec118568b73699e9b314d140",
            "a3bededd14b030fdf06562f6739f879838902f6f4691817573a98bbe9ac6cf7c",
            recipe_digest,
        ):
            self.assertIn(digest, capture)
        self.assertIn("qwen-quant-build-record.json", capture)
        self.assertIn("(record.get(\"output\") or {}).get(\"shards\")", capture)
        self.assertIn("names[0]", capture)
        self.assertIn("quantized_sha256", capture)
        self.assertIn("dd if=\"$1\" iflag=direct", capture)
        self.assertIn("qwen-capture-corpus-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT", capture)
        self.assertIn("docker run --rm --network none", capture)
        self.assertIn("artifact entry is not a regular file", capture)
        self.assertIn("os.chmod(path, 0o400, follow_symlinks=False)", capture)
        self.assertIn("Remove the run-scoped corpus copy", capture)
        self.assertIn("scripts/qwen_capture_control.py", capture)
        self.assertIn("--image-digest \"$QWEN_DEV_IMAGE_DIGEST\"", capture)
        self.assertIn("--mtp-sha256 \"$QWEN_MTP_SHA256\"", capture)
        self.assertIn("test ! -e \"$output\" && test ! -L \"$output\"", capture)
        self.assertIn("$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT", capture)
        self.assertIn("publishes", capture)
        self.assertIn("len(manifest[\"interventions\"]) == 16", capture)
        self.assertIn("QWEN_CAPTURE_MANIFEST_SHA256", capture)
        self.assertIn("Capture manifest SHA-256:", capture)
        self.assertIn("Stock build record SHA-256:", capture)
        for command in (
            "ember-gpu-lock release", "ember-cert-production unmask",
            "ember-cert-production start",
        ):
            self.assertIn(command, capture)
        self.assertIn("if: ${{ always() && steps.qwen-capture-safety", capture)
        self.assertIn("^/qwen-capture-control-[0-9]+$", capture)
        self.assertIn(".gpu-lock-held", capture)
        self.assertIn(".production-masked", capture)
        self.assertIn(".production-was-active", capture)
        self.assertIn("Remove temporary registry credentials", capture)
        for forbidden in (
            "qwen_quantize.py", "docker push", "gh release", "huggingface-cli",
            "actions/upload-artifact", "--execute",
        ):
            self.assertNotIn(forbidden, capture)

    def test_release_build_caches_are_persistent_and_bounded(self) -> None:
        dockerfile = DOCKERFILE.read_text()
        forgejo_ci = FORGEJO_CI.read_text()
        forgejo_container = CONTAINER_WORKFLOW.read_text()
        self.assertIn("ccache", dockerfile)
        self.assertIn("id=ember-gfx1151-ccache", dockerfile)
        self.assertIn("CCACHE_MAXSIZE=20G", dockerfile)
        self.assertIn("ember-ci-ccache", forgejo_ci)
        self.assertIn("CMAKE_C_COMPILER_LAUNCHER=ccache", forgejo_ci)
        self.assertIn("node:24-bookworm@sha256:", forgejo_ci)
        self.assertIn("cancel-in-progress: true", forgejo_ci)
        self.assertNotIn("\n  sanitizers:", forgejo_ci)
        self.assertGreaterEqual(forgejo_ci.count("otheru-forgejo-root.crt:ro"), 4)
        self.assertIn("ember-trivy-cache", forgejo_container)
        self.assertGreaterEqual(GITHUB_CI.read_text().count("ccache --show-stats"), 3)
        self.assertGreaterEqual(forgejo_ci.count("ccache --show-stats"), 2)

    def test_runtime_collector_copies_recursive_elf_closure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                ["bash", str(COLLECT_RUNTIME), directory, "/bin/echo"],
                check=True,
            )
            self.assertTrue((pathlib.Path(directory) / "bin" / "echo").is_file())
            self.assertFalse((pathlib.Path(directory) / "lib").exists())
            self.assertTrue((pathlib.Path(directory) / "usr" / "lib").is_dir())
            copied = [path for path in pathlib.Path(directory).rglob("*") if path.is_file()]
            self.assertGreater(len(copied), 1)

    def test_entrypoint_verifies_model_and_forwards_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = pathlib.Path(directory) / (
                "DeepSeek-V4-Flash-0731-Abliterated-ROCMFPx-Strix-Lean-2.58bpw.gguf"
            )
            draft = pathlib.Path(directory) / (
                "DeepSeek-V4-Flash-0731-Abliterated-DSpark-draft-4.25bpw.gguf"
            )
            model.write_bytes(b"release-test-model")
            draft.write_bytes(b"release-test-draft")
            fake_sha256sum = pathlib.Path(directory) / "sha256sum"
            fake_sha256sum.write_text("#!/usr/bin/env bash\nexit 0\n")
            fake_sha256sum.chmod(0o755)
            env = os.environ | {
                "EMBER_SKIP_DEVICE_CHECK": "1",
                "EMBER_MODEL_DIR": directory,
                "EMBER_SERVER_BIN": "/bin/echo",
                "EMBER_KV_CACHE_DIR": directory,
                "EMBER_HOST": "0.0.0.0",
                "EMBER_PORT": "18080",
                "EMBER_TOOL_LOOP_REPORT": "9",
                "EMBER_NO_PROGRESS_REPORT": "10",
                "EMBER_AUTO_ANSWER_AFTER_LOOP": "11",
                "PATH": directory + os.pathsep + os.environ["PATH"],
            }
            result = subprocess.run(
                ["bash", str(ENTRYPOINT), "--ctx", "42"],
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("-m " + str(model), result.stdout)
            self.assertIn("--host 0.0.0.0", result.stdout)
            self.assertIn("--port 18080", result.stdout)
            self.assertIn("--tool-loop-report 9", result.stdout)
            self.assertIn("--no-progress-report 10", result.stdout)
            self.assertIn("--auto-answer-after-loop 11", result.stdout)
            self.assertIn("--ctx 42", result.stdout)

    def test_qwen_mode_verifies_sealed_shards_and_exports_companions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            env, paths = self.qwen_deployment_fixture(root)
            fake_server = root / "fake-server"
            fake_server.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'mtp=%s\\n' \"$DFLASH_QWEN_MTP\"\n"
                "printf 'depth=%s\\n' \"$DFLASH_QWEN_MTP_DEPTH\"\n"
                "printf 'mmproj=%s\\n' \"$DFLASH_QWEN_VISION_MMPROJ\"\n"
                "printf 'vision_vocab=%s\\n' \"$DFLASH_QWEN_VISION_TEXT_MODEL\"\n"
                "printf 'provider=%s\\n' \"$DFLASH_QWEN_VISION_PROVIDER\"\n"
                "printf 'argv='; printf ' <%s>' \"$@\"; printf '\\n'\n"
            )
            fake_server.chmod(0o755)
            env |= {"EMBER_SERVER_BIN": str(fake_server)}
            result = subprocess.run(
                ["bash", str(ENTRYPOINT), "--ctx", "262144"],
                env=os.environ | env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("sealed Qwen3.8-Flash-Next artifact set verified", result.stdout)
            self.assertIn(f"mtp={paths['mtp']}", result.stdout)
            self.assertIn("depth=3", result.stdout)
            self.assertIn(f"mmproj={paths['mmproj']}", result.stdout)
            self.assertIn(f"vision_vocab={paths['vision_vocab']}", result.stdout)
            self.assertIn(f"provider={paths['provider']}", result.stdout)
            self.assertIn(f" <-m> <{paths['model']}>", result.stdout)
            self.assertIn(" <--ctx> <262144>", result.stdout)

            repeated = subprocess.run(
                ["bash", str(ENTRYPOINT), "--ctx", "262144"],
                env=os.environ | env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertEqual(repeated.stdout.count("integrity cache hit"), 5)

    def test_qwen_mode_rejects_unsealed_or_incomplete_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            env, paths = self.qwen_deployment_fixture(root)
            env["EMBER_SERVER_BIN"] = "/bin/true"
            env["EMBER_VERIFY_EXISTING_SHA256"] = "0"

            paths["model_2"].write_bytes(b"tampered-second-shard")
            result = subprocess.run(
                ["bash", str(ENTRYPOINT)], env=os.environ | env,
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 78)
            self.assertIn("Qwen checksummed artifact SHA-256 mismatch", result.stderr)

            env, paths = self.qwen_deployment_fixture(root)
            lines = paths["checksums"].read_text().splitlines(keepends=True)
            paths["checksums"].write_text("".join(
                line for line in lines if paths["mtp"].name not in line
            ))
            env["EMBER_QWEN_SHA256SUMS_SHA256"] = hashlib.sha256(
                paths["checksums"].read_bytes()
            ).hexdigest()
            env["EMBER_SERVER_BIN"] = "/bin/true"
            result = subprocess.run(
                ["bash", str(ENTRYPOINT)], env=os.environ | env,
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 78)
            self.assertIn("omits the selected MTP companion", result.stderr)

    def test_qwen_mode_requires_first_shard_depth_and_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            env, paths = self.qwen_deployment_fixture(root)
            env["EMBER_SERVER_BIN"] = "/bin/true"
            cases = (
                ("EMBER_QWEN_MODEL", str(paths["model_2"]), "must name shard 00001"),
                ("DFLASH_QWEN_MTP_DEPTH", "0", "integer from 1 to 4"),
                ("DFLASH_QWEN_VISION_PROVIDER", "", "is required"),
            )
            for key, value, message in cases:
                with self.subTest(key=key):
                    changed = env | {key: value}
                    result = subprocess.run(
                        ["bash", str(ENTRYPOINT)], env=os.environ | changed,
                        text=True, capture_output=True,
                    )
                    self.assertEqual(result.returncode, 78)
                    self.assertIn(message, result.stderr)

    def test_entrypoint_rejects_unknown_deployment_mode(self) -> None:
        result = subprocess.run(
            ["bash", str(ENTRYPOINT)],
            env=os.environ | {
                "EMBER_SKIP_DEVICE_CHECK": "1",
                "EMBER_DEPLOYMENT_MODE": "auto",
            },
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 78)
        self.assertIn("must be deepseek-v4-flash or qwen3.8-flash-next", result.stderr)

    def test_default_start_downloads_both_pinned_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            curl_log = root / "curl.log"
            sha256_log = root / "sha256.log"
            fake_curl = root / "curl"
            fake_curl.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "output=\n"
                "url=${@: -1}\n"
                "while (( $# )); do\n"
                "  if [[ $1 == --output ]]; then output=$2; shift 2; else shift; fi\n"
                "done\n"
                "case $url in\n"
                "  *DSpark-draft*) size=10897111840 ;;\n"
                "  *) size=91547243200 ;;\n"
                "esac\n"
                "truncate -s $size \"$output\"\n"
                f"printf '%s\\n' \"$url\" >> {curl_log}\n"
            )
            fake_curl.chmod(0o755)
            fake_sha256sum = root / "sha256sum"
            fake_sha256sum.write_text(
                "#!/usr/bin/env bash\n"
                f"cat >> {sha256_log}\n"
                "exit 0\n"
            )
            fake_sha256sum.chmod(0o755)
            fake_df = root / "df"
            fake_df.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'Filesystem 1-blocks Used Available Use%% Mounted on\\n'\n"
                "printf 'test 300000000000 0 300000000000 0%% /models\\n'\n"
            )
            fake_df.chmod(0o755)
            env = os.environ | {
                "EMBER_SKIP_DEVICE_CHECK": "1",
                "EMBER_MODEL_DIR": directory,
                "EMBER_SERVER_BIN": "/bin/echo",
                "EMBER_KV_CACHE_DIR": directory,
                "EMBER_SEGVTRACE": "",
                # This switch skips only pre-existing files. Both downloads
                # must still cross the digest gate before atomic promotion.
                "EMBER_VERIFY_EXISTING_SHA256": "0",
                "PATH": directory + os.pathsep + os.environ["PATH"],
            }
            subprocess.run(
                ["bash", str(ENTRYPOINT)], env=env, text=True,
                capture_output=True, check=True,
            )
            urls = curl_log.read_text().splitlines()
            self.assertEqual(len(urls), 2)
            self.assertTrue(all(
                "/resolve/9fe32d8d4a1abed16c84e2636b26950232869929/" in url
                for url in urls
            ))
            self.assertTrue(any("ROCMFPx-Strix-Lean-2.58bpw.gguf" in url for url in urls))
            self.assertTrue(any("DSpark-draft-4.25bpw.gguf" in url for url in urls))
            checks = sha256_log.read_text().splitlines()
            self.assertEqual(len(checks), 2)
            self.assertTrue(all(".part" in check for check in checks))
            self.assertEqual(
                len(list((root / "artifact-integrity-v1").glob("*.identity"))), 2
            )

    def test_entrypoint_can_skip_preexisting_artifact_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            model = root / (
                "DeepSeek-V4-Flash-0731-Abliterated-ROCMFPx-Strix-Lean-2.58bpw.gguf"
            )
            draft = root / (
                "DeepSeek-V4-Flash-0731-Abliterated-DSpark-draft-4.25bpw.gguf"
            )
            model.write_bytes(b"pre-provisioned-model")
            draft.write_bytes(b"pre-provisioned-draft")
            marker = root / "sha256-was-called"
            fake_sha256sum = root / "sha256sum"
            fake_sha256sum.write_text(
                "#!/usr/bin/env bash\n"
                f"touch {marker}\n"
                "exit 1\n"
            )
            fake_sha256sum.chmod(0o755)
            env = os.environ | {
                "EMBER_SKIP_DEVICE_CHECK": "1",
                "EMBER_MODEL_DIR": directory,
                "EMBER_SERVER_BIN": "/bin/true",
                "EMBER_KV_CACHE_DIR": directory,
                "EMBER_VERIFY_EXISTING_SHA256": "0",
                "PATH": directory + os.pathsep + os.environ["PATH"],
            }
            result = subprocess.run(
                ["bash", str(ENTRYPOINT)],
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertFalse(marker.exists())
            self.assertEqual(result.stderr.count("WARNING: skipping SHA-256"), 2)

    def test_entrypoint_reuses_identity_bound_integrity_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            model = root / (
                "DeepSeek-V4-Flash-0731-Abliterated-ROCMFPx-Strix-Lean-2.58bpw.gguf"
            )
            draft = root / (
                "DeepSeek-V4-Flash-0731-Abliterated-DSpark-draft-4.25bpw.gguf"
            )
            model.write_bytes(b"pre-provisioned-model")
            draft.write_bytes(b"pre-provisioned-draft")
            sha256_log = root / "sha256.log"
            fake_sha256sum = root / "sha256sum"
            fake_sha256sum.write_text(
                "#!/usr/bin/env bash\n"
                f"cat >> {sha256_log}\n"
                "exit 0\n"
            )
            fake_sha256sum.chmod(0o755)
            env = os.environ | {
                "EMBER_SKIP_DEVICE_CHECK": "1",
                "EMBER_MODEL_DIR": directory,
                "EMBER_SERVER_BIN": "/bin/true",
                "EMBER_KV_CACHE_DIR": directory,
                "PATH": directory + os.pathsep + os.environ["PATH"],
            }

            first = subprocess.run(
                ["bash", str(ENTRYPOINT)], env=env, text=True,
                capture_output=True, check=True,
            )
            self.assertEqual(first.stdout.count("verifying SHA-256"), 2)
            self.assertEqual(len(sha256_log.read_text().splitlines()), 2)

            second = subprocess.run(
                ["bash", str(ENTRYPOINT)], env=env, text=True,
                capture_output=True, check=True,
            )
            self.assertEqual(second.stdout.count("integrity cache hit"), 2)
            self.assertEqual(len(sha256_log.read_text().splitlines()), 2)

            draft.write_bytes(b"changed-draft-with-new-identity")
            third = subprocess.run(
                ["bash", str(ENTRYPOINT)], env=env, text=True,
                capture_output=True, check=True,
            )
            self.assertEqual(third.stdout.count("integrity cache hit"), 1)
            self.assertEqual(third.stdout.count("verifying SHA-256"), 1)
            self.assertEqual(len(sha256_log.read_text().splitlines()), 3)

    def test_entrypoint_rejects_wrong_digest_even_with_ignored_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = pathlib.Path(directory) / (
                "DeepSeek-V4-Flash-0731-Abliterated-ROCMFPx-Strix-Lean-2.58bpw.gguf"
            )
            model.write_bytes(b"tampered")
            env = os.environ | {
                "EMBER_SKIP_DEVICE_CHECK": "1",
                "EMBER_MODEL_DIR": directory,
                "EMBER_MODEL_SHA256": "",
                "EMBER_SERVER_BIN": "/bin/true",
            }
            result = subprocess.run(
                ["bash", str(ENTRYPOINT)], env=env, text=True, capture_output=True
            )
            self.assertEqual(result.returncode, 78)
            self.assertIn("SHA-256 mismatch", result.stderr)

    def test_entrypoint_rejects_invalid_existing_checksum_setting(self) -> None:
        result = subprocess.run(
            ["bash", str(ENTRYPOINT)],
            env=os.environ
            | {
                "EMBER_SKIP_DEVICE_CHECK": "1",
                "EMBER_VERIFY_EXISTING_SHA256": "sometimes",
            },
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 78)
        self.assertIn("must be 0 or 1", result.stderr)

    def test_draft_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = pathlib.Path(directory) / (
                "DeepSeek-V4-Flash-0731-Abliterated-ROCMFPx-Strix-Lean-2.58bpw.gguf"
            )
            model.write_bytes(b"test-model")
            fake_sha256sum = pathlib.Path(directory) / "sha256sum"
            fake_sha256sum.write_text("#!/usr/bin/env bash\nexit 0\n")
            fake_sha256sum.chmod(0o755)
            env = os.environ | {
                "EMBER_SKIP_DEVICE_CHECK": "1",
                "EMBER_MODEL_DIR": directory,
                "EMBER_SERVER_BIN": "/bin/true",
                "EMBER_AUTO_DOWNLOAD": "0",
                "PATH": directory + os.pathsep + os.environ["PATH"],
            }
            result = subprocess.run(
                ["bash", str(ENTRYPOINT)], env=env, text=True, capture_output=True
            )
            self.assertEqual(result.returncode, 66)
            self.assertIn("draft model is not readable", result.stderr)

    def test_entrypoint_preflight_mode_needs_no_model(self) -> None:
        result = subprocess.run(
            ["bash", str(ENTRYPOINT)],
            env=os.environ
            | {"EMBER_SKIP_DEVICE_CHECK": "1", "EMBER_PREFLIGHT_ONLY": "1"},
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("preflight passed", result.stdout)

    def test_smoke_test_checks_read_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_curl = pathlib.Path(directory) / "curl"
            fake_curl.write_text(
                "#!/usr/bin/env bash\n"
                "case \"${*: -1}\" in\n"
                "  */health) printf '{\"status\":\"ok\"}\\n' ;;\n"
                "  */status) printf '{\"busy\":false}\\n' ;;\n"
                "  */v1/models) printf '{\"object\":\"list\",\"data\":[{\"id\":\"deepseek-v4-flash\"}]}\\n' ;;\n"
                "  *) exit 22 ;;\n"
                "esac\n"
            )
            fake_curl.chmod(0o755)
            env = os.environ | {"PATH": directory + os.pathsep + os.environ["PATH"]}
            result = subprocess.run(
                ["bash", str(SMOKE)],
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("smoke test passed", result.stdout)


if __name__ == "__main__":
    unittest.main()


class ToolResultDecodePolicyTest(unittest.TestCase):
    """The escape hatch that lets the tool-result rule be measured.

    Speculation is withheld from every turn that follows a tool result, because
    speculative verification was once seen to change a near-tied token and
    re-emit an already-successful write_file call. EMBER_TOOL_RESULT_AR=0 lifts
    that so the case can be measured; it must never become the default by
    accident, since the rule guards a real incident.
    """

    def test_rule_is_on_unless_explicitly_disabled(self) -> None:
        src = SERVER_MAIN.read_text()
        self.assertIn("EMBER_TOOL_RESULT_AR", src)
        # Only the literal "0" disables it: an unset or malformed value keeps
        # today's behaviour rather than silently lifting the rule.
        self.assertIn('cached = (e && e[0] == \'0\') ? 0 : 1;', src)
        self.assertIn("tool_result_forces_ar() &&", src)
        self.assertIn(
            "ember_chat_request_is_tool_result_continuation(req);", src)
