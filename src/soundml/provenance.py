"""Run record —— 每次跑批留一条可举证的痕迹。

每一条结论都要能指回产生它的 run_id。审稿人要求复现时，这里就是全部证据。

一条 run record 包含：
  - run_id / 时间戳
  - git commit + 工作区是否干净
  - 随机种子
  - 数据快照 SHA256（输入文件集合的内容哈希，不是路径哈希）
  - 特征提取器版本号
  - 完整超参
  - 依赖锁定文件哈希（uv.lock）
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def snapshot_hash(paths: list[Path]) -> dict[str, Any]:
    """对一组文件做内容级快照哈希。

    先按相对路径排序，再把 "相对路径:文件哈希" 逐条喂进总哈希——
    这样文件改名、增删、内容变动都会改变结果，而磁盘位置变化不会。
    """
    entries = []
    roll = hashlib.sha256()
    for p in sorted(paths, key=lambda x: str(x).replace("\\", "/")):
        digest = sha256_file(p)
        try:
            rel = p.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            rel = p.name
        roll.update(f"{rel}:{digest}\n".encode())
        entries.append({"path": rel, "sha256": digest})
    return {"n_files": len(entries), "rollup_sha256": roll.hexdigest(), "files": entries}


class RunRecord:
    """用法::

        with RunRecord("train_baseline", seed=42, params={...}) as run:
            run.add_input_snapshot("labels", [labels_csv])
            ...
            run.log_metric("auc_groupcv", 0.98)
            run.save_artifact("shap_summary.png", fig_path)
    """

    def __init__(self, name: str, seed: int, params: dict[str, Any] | None = None):
        started = datetime.now(timezone.utc)
        self.run_id = f"{started:%Y%m%dT%H%M%SZ}_{name}"
        self.name = name
        self.dir = RUNS_DIR / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)

        lock = REPO_ROOT / "uv.lock"
        self.meta: dict[str, Any] = {
            "run_id": self.run_id,
            "name": name,
            "started_utc": started.isoformat(),
            "seed": seed,
            "params": params or {},
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "uv_lock_sha256": sha256_file(lock) if lock.exists() else None,
            "inputs": {},
            "metrics": {},
            "artifacts": [],
        }

    def add_input_snapshot(self, key: str, paths: list[Path]) -> None:
        self.meta["inputs"][key] = snapshot_hash(paths)

    def note(self, key: str, value: Any) -> None:
        self.meta.setdefault("notes", {})[key] = value

    def log_metric(self, key: str, value: Any) -> None:
        self.meta["metrics"][key] = value

    def artifact_path(self, filename: str) -> Path:
        self.meta["artifacts"].append(filename)
        return self.dir / filename

    def write(self) -> Path:
        self.meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        path = self.dir / "meta.json"
        path.write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def __enter__(self) -> RunRecord:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.meta["failed"] = f"{exc_type.__name__}: {exc}"
        self.write()
