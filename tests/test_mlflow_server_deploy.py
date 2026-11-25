"""tools/mlflow-server: the deployment's safety-relevant settings stay as decided (no server needed)."""
from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

import pytest

D = Path(__file__).resolve().parents[1] / "tools" / "mlflow-server"
yaml = pytest.importorskip("yaml")


def test_mlflow_is_pinned_to_the_helpers_minor_and_locked():
    deps = tomllib.loads((D / "pyproject.toml").read_text())["project"]["dependencies"]
    assert "mlflow[auth]==3.18.0" in deps
    lock = (D / "uv.lock").read_text()
    assert re.search(r'name = "mlflow"\nversion = "3\.18\.0"', lock)
    assert re.search(r'name = "psycopg"\nversion = "3\.', lock)


def test_postgres_is_pinned_local_only_and_persistent():
    pg = yaml.safe_load((D / "docker-compose.yml").read_text())["services"]["postgres"]
    assert re.fullmatch(r"postgres:18@sha256:[0-9a-f]{64}", pg["image"])
    assert all(p.startswith("127.0.0.1:") for p in pg["ports"])
    assert pg["restart"] == "unless-stopped"
    assert any(v.startswith("iris-mlflow-pgdata:") for v in pg["volumes"])
    assert "secrets/postgres.env" in pg["env_file"] and "POSTGRES_PASSWORD" not in pg.get("environment", {})


def test_the_server_refuses_without_a_good_mount():
    s = (D / "serverctl.sh").read_text()
    server = s[s.index("do_server() {"):s.index("stop_server() {")]
    assert server.index("do_check") < server.index('"$VENV/bin/mlflow" server')     # checked right before starting
    check = s[s.index("check_mount() {"):s.index("do_check() {")]
    assert 'mountpoint -q "$MOUNT"' in check and 'rm -f "$probe"' in check and 'cat "$probe"' in check
    assert '--artifacts-destination "$STORE"' in s and 'STORE="$MOUNT/store"' in s
    assert "--serve-artifacts" in s and "--app-name basic-auth" in s


def test_no_secret_on_a_command_line_or_in_the_repo():
    s = (D / "serverctl.sh").read_text()
    assert "postgresql://mlflow@127.0.0.1" in s and not re.search(r"postgresql://mlflow:[^@]+@", s)   # pgpass
    assert not re.search(r"curl[^\n]* -u ", s)                                      # credentials via `curl -K -`
    tracked = subprocess.run(["git", "ls-files", str(D)], capture_output=True, text=True, cwd=D).stdout
    assert not re.search(r"(secrets/|\.env$|pgpass|basic_auth\.ini)", tracked, re.M)


@pytest.mark.parametrize("script", ["serverctl.sh", "backup.sh", "restore_test.sh", "install.sh"])
def test_scripts_parse(script):
    assert subprocess.run(["bash", "-n", str(D / script)]).returncode == 0


def test_the_mount_reports_uploads_over_a_private_unix_socket():
    s = (D / "serverctl.sh").read_text()
    mount = s[s.index("do_mount() {"):s.index("check_mount() {")]
    assert '--rc --rc-addr "unix://$RCSOCK"' in mount and 'chmod 700 "$RCDIR"' in mount   # vlm only, no TCP port
    assert 'RCSOCK="$RCDIR/' in s and "--rc-no-auth" not in s and "--rc-addr localhost" not in s
    status = s[s.index("upload_status() {"):s.index("with_lock() {")]
    assert "vfs/stats" in status and "uploadsInProgress" in status and "uploadsQueued" in status
    assert '"Dirty"' in status                                   # fallback for a mount started without --rc
    assert 'du -sh "$CACHE"' in status


def test_the_sharepoint_folders_live_in_repo_data_and_agree():
    """Artifacts and backups sit beside dvc-store in IRIS/repo-data; every script names the same two folders."""
    art, bak = "iris-onedrive:IRIS/repo-data/mlflow-artifacts", "iris-onedrive:IRIS/repo-data/mlflow-backups"
    text = {f: (D / f).read_text() for f in ("serverctl.sh", "backup.sh", "restore_test.sh", "install.sh")}
    assert re.search(rf"^REMOTE={re.escape(art)}$", text["serverctl.sh"], re.M)
    assert re.search(rf"^DEST={re.escape(bak)}$", text["backup.sh"], re.M)
    assert re.search(rf"^SRC={re.escape(bak)}$", text["restore_test.sh"], re.M)
    assert f'mkdir {art}/store\n' in text["install.sh"] and f'mkdir {bak}\n' in text["install.sh"]
    for f, s in [*text.items(), ("README.md", (D / "README.md").read_text())]:
        assert not re.search(r"IRIS/mlflow-(artifacts|backups)", s), f"{f} still names the old folder"


def _run_backup_with_fake_rclone(tmp_path, lsf_stdout, lsf_rc=0):
    """backup.sh against a fake rclone (config.env overrides RCLONE) that logs every call."""
    calls = tmp_path / "calls"
    fake = tmp_path / "rclone"
    fake.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> {calls}\n'
        f'if [ "$1" = lsf ]; then printf %s "{lsf_stdout}"; exit {lsf_rc}; fi\n'
        "exit 0\n"
    )
    fake.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.env").write_text(f"RCLONE={fake}\n")
    r = subprocess.run(["bash", str(D / "backup.sh")], capture_output=True, text=True,
                       env={"IRIS_MLFLOW_HOME": str(home), "HOME": str(tmp_path), "PATH": "/usr/bin:/bin"})
    return r, (calls.read_text() if calls.exists() else "")


@pytest.mark.parametrize("lsf_stdout,lsf_rc", [("dvc-store/\\nmlflow-artifacts/\\n", 0), ("", 1)],
                         ids=["folder-missing", "listing-fails"])
def test_backup_refuses_without_the_backup_folder_and_never_creates_it(tmp_path, lsf_stdout, lsf_rc):
    r, calls = _run_backup_with_fake_rclone(tmp_path, lsf_stdout, lsf_rc)
    assert r.returncode != 0 and "REFUSED" in r.stderr
    assert calls.splitlines() == ["lsf --dirs-only --max-depth 1 iris-onedrive:IRIS/repo-data"]   # no mkdir, no copy
    assert not list((tmp_path / "home" / "backups").iterdir())                                  # no dump taken


def test_backup_never_makes_its_folder_and_checks_before_every_upload():
    s = (D / "backup.sh").read_text()
    assert not re.search(r'"\$RCLONE" (mkdir|mkdirs)\b', s)
    loop = s[s.index("for db in"):]
    assert loop.index("require_dest") < loop.index('"$RCLONE" copyto')
