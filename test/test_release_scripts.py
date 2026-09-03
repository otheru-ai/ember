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
BUILD_SH = ROOT / "scripts" / "build.sh"
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
            "2ff6ff0c4bd20d8438113404d9c7c3d4495bbc4b43b5622f37a0f68aebfebbc2",
            script,
        )
        self.assertIn(
            "1a01c80eceae302bcc1d70836759ee97974d7983c5084ef43f6ef772a8970ae6",
            script,
        )
        self.assertIn("4b551c949d44137efc8b615c6c015a6ce677d9a2", script)
        self.assertNotIn("EMBER_MODEL:-", script)
        self.assertNotIn("EMBER_DRAFT_MODEL:-", script)
        self.assertNotIn("EMBER_MODEL_SHA256-", script)
        self.assertNotIn("EMBER_DRAFT_MODEL_SHA256-", script)

    def test_shell_syntax(self) -> None:
        for script in (
            BUILD_SH,
            ENTRYPOINT,
            COLLECT_RUNTIME,
            PREFLIGHT,
            SMOKE,
        ):
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_rocm_build_binds_revision_before_entering_container(self) -> None:
        script = BUILD_SH.read_text()
        self.assertIn('REVISION="$(git -C "$REPO" rev-parse HEAD)"', script)
        self.assertIn('[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]]', script)
        self.assertIn('-DEMBER_CONFIGURED_GIT_HEAD="${REVISION}"', script)
        self.assertIn(
            "--target ember-dflash ember-gguf-quantize -j ${JOBS}", script)

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
        # the vision model under test
        self.assertIn(
            "2ff6ff0c4bd20d8438113404d9c7c3d4495bbc4b43b5622f37a0f68aebfebbc2",
            certify,
        )
        # its tower: certification must exercise the path the default
        # deployment serves, not text alone
        self.assertIn(
            "9225c5562c05bd910245ab24c9274ca777eba2a801990f47ebe0c6344f144002",
            certify,
        )
        self.assertIn("--vision-mmproj", certify)
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
        # It authenticates and pulls the published candidate rather than
        # building anything on the machine that holds production.
        self.assertIn("GHCR_TOKEN: ${{ secrets.GITHUB_TOKEN }}", certify_job)
        self.assertIn("docker login ghcr.io", certify_job)
        self.assertIn("docker pull", certify_job)
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

    def test_certify_mounts_match_the_entrypoint_filenames(self) -> None:
        """The smoke test mounts artifacts at the names the entrypoint expects.

        These live in two files and drifted once already: the default model
        moved to the vision artifact while the certify mount kept the old 0731
        name, so the entrypoint could not find its model and downloaded 85 GiB
        instead, which consumed the step and the run. Asserting the names
        against each other is what makes that a red test rather than a wasted
        certification.
        """
        certify = GITHUB_CERTIFY.read_text()
        entrypoint = ENTRYPOINT.read_text()

        def entrypoint_value(name: str) -> str:
            match = re.search(rf'^{name}="([^"]+)"', entrypoint, re.MULTILINE)
            self.assertIsNotNone(match, f"{name} not found in the entrypoint")
            return match.group(1)

        for var in ("file", "draft_file", "mmproj_file"):
            self.assertIn(f"/models/{entrypoint_value(var)}", certify,
                          f"certify does not mount the entrypoint's {var}")

        # And it must never resolve a missing artifact by fetching one: a
        # downloaded replacement would certify something other than what is on
        # the box.
        self.assertIn("EMBER_AUTO_DOWNLOAD=0", certify)

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
                "DeepSeek-V4-Flash-Vision-Exp-Abliterated-ROCMFPx-Strix-Lean-2.58bpw.gguf"
            )
            draft = pathlib.Path(directory) / (
                "DeepSeek-V4-Flash-0731-Abliterated-DSpark-draft-4.25bpw.gguf"
            )
            model.write_bytes(b"release-test-model")
            draft.write_bytes(b"release-test-draft")
            mmproj = pathlib.Path(directory) / "mmproj-DeepSeek-V4-Flash-Vision-Exp-F16.gguf"
            mmproj.write_bytes(b"release-test-mmproj")
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
        self.assertIn("must be deepseek-v4-flash", result.stderr)

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
            # model, drafter and vision tower
            self.assertEqual(len(urls), 3)
            self.assertTrue(all(
                "/resolve/4b551c949d44137efc8b615c6c015a6ce677d9a2/" in url
                for url in urls
            ))
            self.assertTrue(any("ROCMFPx-Strix-Lean-2.58bpw.gguf" in url for url in urls))
            self.assertTrue(any("DSpark-draft-4.25bpw.gguf" in url for url in urls))
            checks = sha256_log.read_text().splitlines()
            self.assertEqual(len(checks), 3)
            self.assertTrue(all(".part" in check for check in checks))
            self.assertEqual(
                len(list((root / "artifact-integrity-v1").glob("*.identity"))), 3
            )

    def test_entrypoint_can_skip_preexisting_artifact_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            model = root / (
                "DeepSeek-V4-Flash-Vision-Exp-Abliterated-ROCMFPx-Strix-Lean-2.58bpw.gguf"
            )
            draft = root / (
                "DeepSeek-V4-Flash-0731-Abliterated-DSpark-draft-4.25bpw.gguf"
            )
            model.write_bytes(b"pre-provisioned-model")
            draft.write_bytes(b"pre-provisioned-draft")
            mmproj = pathlib.Path(directory) / "mmproj-DeepSeek-V4-Flash-Vision-Exp-F16.gguf"
            mmproj.write_bytes(b"mmproj")
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
            self.assertEqual(result.stderr.count("WARNING: skipping SHA-256"), 3)

    def test_entrypoint_reuses_identity_bound_integrity_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            model = root / (
                "DeepSeek-V4-Flash-Vision-Exp-Abliterated-ROCMFPx-Strix-Lean-2.58bpw.gguf"
            )
            draft = root / (
                "DeepSeek-V4-Flash-0731-Abliterated-DSpark-draft-4.25bpw.gguf"
            )
            model.write_bytes(b"pre-provisioned-model")
            draft.write_bytes(b"pre-provisioned-draft")
            mmproj = pathlib.Path(directory) / "mmproj-DeepSeek-V4-Flash-Vision-Exp-F16.gguf"
            mmproj.write_bytes(b"mmproj")
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
            self.assertEqual(first.stdout.count("verifying SHA-256"), 3)
            self.assertEqual(len(sha256_log.read_text().splitlines()), 3)

            second = subprocess.run(
                ["bash", str(ENTRYPOINT)], env=env, text=True,
                capture_output=True, check=True,
            )
            self.assertEqual(second.stdout.count("integrity cache hit"), 3)
            self.assertEqual(len(sha256_log.read_text().splitlines()), 3)

            # Only the drafter changes here: the model and the tower keep their
            # identity records, so exactly one artifact is re-verified.
            draft.write_bytes(b"changed-draft-with-new-identity")
            third = subprocess.run(
                ["bash", str(ENTRYPOINT)], env=env, text=True,
                capture_output=True, check=True,
            )
            self.assertEqual(third.stdout.count("integrity cache hit"), 2)
            self.assertEqual(third.stdout.count("verifying SHA-256"), 1)
            self.assertEqual(len(sha256_log.read_text().splitlines()), 4)

    def test_entrypoint_rejects_wrong_digest_even_with_ignored_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = pathlib.Path(directory) / (
                "DeepSeek-V4-Flash-Vision-Exp-Abliterated-ROCMFPx-Strix-Lean-2.58bpw.gguf"
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
                "DeepSeek-V4-Flash-Vision-Exp-Abliterated-ROCMFPx-Strix-Lean-2.58bpw.gguf"
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
        self.assertIn("ember_chat_request_is_tool_result_continuation(req)", src)
        # An image request forces autoregressive decode too: the vision path
        # declines speculation rather than degrading it silently.
        self.assertIn("greq.force_ar_decode = req->has_images ||", src)
