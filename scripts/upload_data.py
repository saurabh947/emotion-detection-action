#!/usr/bin/env python3
"""Stream large dataset to a remote VM via a single SF Compute SSH connection.

Works by opening ONE `sf nodes ssh` subprocess, sending a Python receiver
command to the remote shell via stdin, then streaming the raw binary tar.gz
data immediately after. The remote Python reads exactly file_size bytes so
the shell can cleanly receive an `exit 0` afterwards.

Usage (called by remote.sh sync-data):
    python3 scripts/upload_data.py <node_name> <local_data_dir> <remote_dir>

Example:
    python3 scripts/upload_data.py my-node ./data/combined /root/emotion-detection-action/data/combined
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


_SAFE_PATTERN = re.compile(r"^[A-Za-z0-9._/~-]+$")


def _assert_safe_path(value: str, name: str) -> None:
    """Reject paths containing shell metacharacters to prevent injection."""
    if not _SAFE_PATTERN.match(value):
        print(
            f"  ERROR: {name} contains unsafe characters: {value!r}\n"
            "  Only alphanumeric, '.', '_', '/', '~', and '-' are allowed.",
            file=sys.stderr,
        )
        sys.exit(1)


def stream_upload(node_name: str, local_data_dir: str, remote_combined_dir: str) -> None:
    _assert_safe_path(node_name, "node_name")
    _assert_safe_path(remote_combined_dir, "remote_combined_dir")

    if not os.path.isdir(local_data_dir):
        print(f"  ERROR: Local data directory not found: {local_data_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Step 1: create tar.gz locally ───────────────────────────────────────
    # Use mkstemp to avoid the TOCTOU race condition of the deprecated mktemp().
    _fd, tmp_tar = tempfile.mkstemp(suffix=".tar.gz", prefix="eda_upload_")
    os.close(_fd)  # Close the file descriptor; tarfile.open will reopen it.
    data_dir = local_data_dir.rstrip("/")
    parent   = os.path.dirname(data_dir)
    basename = os.path.basename(data_dir)

    print(f"  Compressing {local_data_dir} …")
    t0 = time.time()
    with tarfile.open(tmp_tar, "w:gz") as tf:
        tf.add(data_dir, arcname=basename)
    elapsed = time.time() - t0
    file_size = os.path.getsize(tmp_tar)
    print(f"  Compressed: {human_size(file_size)}  ({elapsed:.0f}s)")

    # ── Step 2: open ONE sf nodes ssh subprocess ─────────────────────────────
    # The remote shell receives commands via stdin (non-interactive mode).
    # We send a Python one-liner that reads exactly file_size raw bytes from
    # stdin (which is the SSH pipe), writes them to a temp file, then extracts.
    # After Python consumes its bytes, the shell reads our final `exit 0\n`.
    #
    # Use a randomized remote temp path to avoid symlink-attack races on /tmp.
    import secrets
    remote_tmp = f"/tmp/_eda_upload_{secrets.token_hex(8)}.tar.gz"
    receiver = (
        f"python3 -c \""
        f"import sys,os,subprocess;"
        f"os.makedirs('{remote_combined_dir}',exist_ok=True);"
        f"data=sys.stdin.buffer.read({file_size});"
        f"open('{remote_tmp}','wb').write(data);"
        f"subprocess.run(['tar','-xzf','{remote_tmp}','--strip-components=1','-C','{remote_combined_dir}']);"
        f"os.remove('{remote_tmp}');"
        f"print('UPLOAD_DONE')\"\n"
    )

    print(f"  Connecting to SF Compute node: {node_name} …")
    proc = subprocess.Popen(
        ["sf", "nodes", "ssh", "-q", f"root@{node_name}"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # ── Step 3: send receiver command ───────────────────────────────────
        proc.stdin.write(receiver.encode())

        # ── Step 4: stream raw binary tar.gz data ───────────────────────────
        chunk_size = 4 * 1024 * 1024  # 4 MB chunks
        sent = 0
        t_start = time.time()
        print(f"  Uploading {human_size(file_size)} …")

        with open(tmp_tar, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                proc.stdin.write(chunk)
                proc.stdin.flush()
                sent += len(chunk)
                pct   = sent / file_size * 100
                speed = sent / max(time.time() - t_start, 0.1)
                eta   = (file_size - sent) / max(speed, 1)
                print(
                    f"\r  {human_size(sent)} / {human_size(file_size)}"
                    f"  ({pct:.0f}%)  {human_size(int(speed))}/s"
                    f"  ETA {eta:.0f}s   ",
                    end="",
                    flush=True,
                )

        print()  # newline after progress

        # ── Step 5: close stdin so Python's read() returns, then exit ───────
        proc.stdin.write(b"\nexit 0\n")
        proc.stdin.close()

        stdout, stderr = proc.communicate()

        # ── Step 6: check result ─────────────────────────────────────────────
        if proc.returncode != 0:
            print(f"\n  ERROR: Upload failed (exit {proc.returncode})", file=sys.stderr)
            if stderr:
                print(stderr.decode(errors="replace"), file=sys.stderr)
            sys.exit(proc.returncode)

        if b"UPLOAD_DONE" not in stdout:
            print("\n  WARNING: Did not receive UPLOAD_DONE confirmation.", file=sys.stderr)
            print("  stdout:", stdout.decode(errors="replace")[:500], file=sys.stderr)
        else:
            elapsed_total = time.time() - t_start
            print(f"  Upload and extraction complete. ({elapsed_total:.0f}s total)")

    finally:
        # Always remove the local temp archive, even on error or KeyboardInterrupt.
        try:
            os.remove(tmp_tar)
        except OSError:
            pass


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <node_name> <local_data_dir> <remote_combined_dir>")
        sys.exit(1)

    stream_upload(
        node_name         = sys.argv[1],
        local_data_dir    = sys.argv[2],
        remote_combined_dir = sys.argv[3],
    )
