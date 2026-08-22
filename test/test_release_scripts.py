#!/usr/bin/env python3
"""GPU-free contract tests for container and operator scripts."""

from __future__ import annotations

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
        self.assertIn("EMBER_HOST: ${EMBER_HOST:-127.0.0.1}", release_service)
        self.assertIn(
            "EMBER_VERIFY_EXISTING_SHA256: ${EMBER_VERIFY_EXISTING_SHA256:-1}",
            release_service,
        )
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
        certify_job = certify.split("\n  promote:", 1)[0]
        self.assertNotIn("actions/checkout", certify_job)
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
        self.assertIn(
            "needs: [invariants, build-test, sanitizers, analyzer, coverage]", ci
        )
        self.assertIn("uses: ./.github/workflows/container.yml", ci)
        self.assertIn("workflow_call:", container)
        self.assertIn('git rev-parse "$publish_sha^"', container)
        self.assertIn("expected=(CHANGELOG.md compose.yaml VERSION)", container)
        self.assertIn("release_tag:", container)
        self.assertIn("inputs.release_tag || github.sha", container)
        self.assertIn("workflow_run:", release_notes)
        self.assertIn("conclusion == 'success'", release_notes)
        self.assertIn("ci/release_changelog.py notes", release_notes)
        self.assertIn("gh release create", release_notes)
        self.assertIn("!startsWith(github.event.head_commit.message", ci)
        self.assertIn("certify-and-release:", ci)
        self.assertIn("needs: publish-candidate", ci)
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
        self.assertGreaterEqual(forgejo_ci.count("otheru-forgejo-root.crt:ro"), 5)
        self.assertIn("ember-trivy-cache", forgejo_container)
        self.assertGreaterEqual(GITHUB_CI.read_text().count("ccache --show-stats"), 3)
        self.assertGreaterEqual(forgejo_ci.count("ccache --show-stats"), 3)

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
