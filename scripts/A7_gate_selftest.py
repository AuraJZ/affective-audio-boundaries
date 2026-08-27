"""阳性对照:确认 A7 的规则表内容闸真的会响。

不会响的检查比没有检查更坏。这次公开发布之所以漏掉三个 run record,
就是因为清理只按文件名做,没有任何东西按内容再查一遍。

    .venv/Scripts/python scripts/A7_gate_selftest.py
"""
import importlib.util
import re
import sys
from pathlib import Path

# 🔴 先导入 A7,再谈 stdout。A7 在模块层把 sys.stdout 换成自己的 TextIOWrapper;
# 如果这里先包装一次,那个 wrapper 会被顶掉并被回收,连带关闭底层缓冲,
# 于是下一行 print 就是 "I/O operation on closed file"。
HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("a7", HERE / "A7_assemble_release.py")
a7 = importlib.util.module_from_spec(spec)
sys.modules["a7"] = a7
spec.loader.exec_module(a7)          # 之后 sys.stdout 已是 UTF-8,不必再包

ROOT = a7.REPO_ROOT
SIG = a7.RULE_TABLE_SIGNATURE
print(f"签名:{SIG}\n")

SKIP = {".git", ".venv", "data", "logs", "tmp", "dist", "__pycache__"}
hits = []
for p in ROOT.rglob("*"):
    if not p.is_file() or p.suffix.lower() not in {".json", ".md", ".csv", ".py", ".txt"}:
        continue
    if any(x in p.parts for x in SKIP) or p.stat().st_size > 4_000_000:
        continue
    try:
        t = p.read_text(encoding="utf-8", errors="replace")
    except Exception:                                        # noqa: BLE001
        continue
    if all(re.search(pat, t) for pat in SIG):
        hits.append(p.relative_to(ROOT).as_posix())

print(f"工作仓库里命中 {len(hits)} 处:")
for h in sorted(hits):
    guarded = a7.excluded(h)
    print(f"  {'名字排除已覆盖' if guarded else '⚠️ 仅靠内容闸拦下'}  {h}")

expect = 3
ok = len(hits) >= expect
print(f"\n{'✅' if ok else '🔴'} 预期至少 {expect} 处(三个 rules_draft 的 meta.json)。"
      f"{'闸门可用。' if ok else '闸门没有响 —— 签名写错了。'}")
sys.exit(0 if ok else 1)
