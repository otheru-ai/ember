#!/usr/bin/env python3
"""GPU-free contract tests for container and operator scripts."""

from __future__ import annotations

import hashlib
import os
import pathlib
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
COMPOSE = ROOT / "compose.yaml"
COMPOSE_BUILD = ROOT / "compose.build.yaml"
PREFLIGHT = ROOT / "scripts" / "preflight.sh"
SMOKE = ROOT / "scripts" / "smoke_test.sh"


class ReleaseScriptTests(unittest.TestCase):
    def test_default_model_is_immutably_pinned(self) -> None:
        script = ENTRYPOINT.read_text()
        self.assertIn("/resolve/$revision/$file", script)
        self.assertNotIn("/resolve/main/$file", script)
        self.assertIn(
            "18aec8c0be4087007e557aa6020b28f12cd4c5d1f9c67b2a815c152aea97b3ed",
            script,
        )

    def test_shell_syntax(self) -> None:
        for script in (ENTRYPOINT, COLLECT_RUNTIME, PREFLIGHT, SMOKE):
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_container_targets_separate_toolchain_from_runtime(self) -> None:
        dockerfile = DOCKERFILE.read_text()
        self.assertIn("AS dev", dockerfile)
        self.assertIn("AS release", dockerfile)
        self.assertGreaterEqual(dockerfile.count("@sha256:"), 2)
        release = dockerfile.split("AS release", 1)[1]
        self.assertNotIn("build-essential", release)
        self.assertNotIn("cmake --build", release)
        self.assertIn("COPY --from=dev /ember-runtime/ /", release)
        self.assertIn("/usr/share/licenses/ember/", release)
        self.assertIn("blas_lib_gfx1151.kpack", dockerfile)
        self.assertIn("rocblas/library/gfx1151", dockerfile)
        self.assertIn("TensileLibrary_lazy_gfx1151.dat", dockerfile)

    def test_compose_pulls_release_and_keeps_source_build_explicit(self) -> None:
        compose = COMPOSE.read_text()
        build = COMPOSE_BUILD.read_text()
        release_service = compose.split("  ember-dev:", 1)[0]
        version = (ROOT / "VERSION").read_text().strip()
        self.assertIn(f"ghcr.io/otheru/ember:{version}", release_service)
        self.assertIn("pull_policy: always", release_service)
        self.assertNotIn("build:", release_service)
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
        self.assertIn("runs-on: ubuntu-latest-8-cores", container)
        self.assertIn("packages: write", container)
        self.assertIn("ghcr.io/${GITHUB_REPOSITORY,,}", container)
        self.assertIn("EMBER_GFX1151_CERTIFIED_SHA", container)
        self.assertIn("runs-on: [self-hosted, linux, x64, gfx1151]", certify)
        self.assertIn("environment: gfx1151-certification", certify)
        self.assertIn("--validate-gemm-batch 64", certify)
        self.assertIn("--validate-prompt", certify)
        self.assertIn("org.opencontainers.image.revision", certify)
        self.assertNotIn("actions/checkout", certify)
        triggers = certify.split("permissions:", 1)[0]
        self.assertNotIn("pull_request", triggers)

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
            model = pathlib.Path(directory) / "model.gguf"
            model.write_bytes(b"release-test-model")
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            env = os.environ | {
                "EMBER_SKIP_DEVICE_CHECK": "1",
                "EMBER_MODEL": str(model),
                "EMBER_MODEL_DIR": directory,
                "EMBER_MODEL_SHA256": digest,
                "EMBER_SERVER_BIN": "/bin/echo",
                "EMBER_KV_CACHE_DIR": directory,
                "EMBER_PORT": "18080",
                "EMBER_TOOL_LOOP_REPORT": "9",
                "EMBER_NO_PROGRESS_REPORT": "10",
                "EMBER_AUTO_ANSWER_AFTER_LOOP": "11",
            }
            result = subprocess.run(
                ["bash", str(ENTRYPOINT), "--ctx", "42"],
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("-m " + str(model), result.stdout)
            self.assertIn("--port 18080", result.stdout)
            self.assertIn("--tool-loop-report 9", result.stdout)
            self.assertIn("--no-progress-report 10", result.stdout)
            self.assertIn("--auto-answer-after-loop 11", result.stdout)
            self.assertIn("--ctx 42", result.stdout)

    def test_entrypoint_rejects_wrong_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = pathlib.Path(directory) / "model.gguf"
            model.write_bytes(b"tampered")
            env = os.environ | {
                "EMBER_SKIP_DEVICE_CHECK": "1",
                "EMBER_MODEL": str(model),
                "EMBER_MODEL_DIR": directory,
                "EMBER_MODEL_SHA256": "0" * 64,
                "EMBER_SERVER_BIN": "/bin/true",
            }
            result = subprocess.run(
                ["bash", str(ENTRYPOINT)], env=env, text=True, capture_output=True
            )
            self.assertEqual(result.returncode, 78)
            self.assertIn("SHA-256 mismatch", result.stderr)

    def test_custom_model_can_explicitly_disable_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = pathlib.Path(directory) / "local.gguf"
            model.write_bytes(b"local-development-model")
            env = os.environ | {
                "EMBER_SKIP_DEVICE_CHECK": "1",
                "EMBER_MODEL": str(model),
                "EMBER_MODEL_DIR": directory,
                "EMBER_MODEL_SHA256": "",
                "EMBER_SERVER_BIN": "/bin/true",
            }
            subprocess.run(["bash", str(ENTRYPOINT)], env=env, check=True)

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
