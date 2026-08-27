r"""核验 催眠_光影_睡眠参数对照表 的声学条目是否指向真实文献。

    .venv/Scripts/python scripts/65_audit_reference_table.py

## 背景

`03-科学研究与参数/催眠_光影_睡眠参数对照表.csv` 早于本次工作三个多月，
非本次产出。它给出 15 条声学结论，每条附 PubMed / DOI 链接。

**不能凭表格自身判断结论是否可靠** —— 参考文献表最常见的失效方式，
正是本项目在建 `references.bib` 时亲历的那种：条目看起来完整、
作者年份期刊都合理，但指向的文献不存在或不相干。当时按标题检索、
阈值 0.6，混进了五篇无关文献。

因此这里用同一套方法逐条核：解析链接 → 取回真实标题与年份 →
与表中记载比对。

## 判据

| 结果 | 含义 |
|---|---|
| ✅ 标题与年份都对上 | 条目可信，结论可按表中记载引用 |
| ⚠️ 年份或标题有出入 | 需人工看原文 |
| ❌ 解析不到 | **不得引用**，需重新检索 |

**本脚本只核「文献存在且是这一篇」，不核「表中的结论概括是否忠实于原文」** ——
后者只能读原文。核不出问题不等于结论正确。
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 🔴 这里原本硬编码了一条指向仓库之外某个私人笔记库的绝对路径。
# 两个问题：它把那个库的目录结构写进了会公开的代码，而且对任何其他人都跑不
# 起来。后来那个目录被改名，路径当场失效 —— 硬编码的私人路径迟早会断，
# 只是不知道哪天。
#
# 现在从环境变量取，取不到就说清楚要什么并干净退出，不抛栈。这张表不在本
# 仓库里，也不该在：它是库里的私人笔记。
TABLE_ENV = "SOUNDML_REFERENCE_TABLE"
TABLE = Path(os.environ[TABLE_ENV]) if os.environ.get(TABLE_ENV) else None
UA = "soundml-audit/1.0 (mailto:research@example.org)"
AUDIO_TAGS = ("sound", "audio", "acoust", "noise", "music", "auditory")


def _get(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            return json.load(r)
    except Exception:                                        # noqa: BLE001
        return None


def by_pmid(pmid: str) -> tuple[str, int] | None:
    d = _get("https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
             + urllib.parse.urlencode({"query": f"EXT_ID:{pmid} AND SRC:MED",
                                       "format": "json", "resultType": "lite"}))
    if not d:
        return None
    res = d.get("resultList", {}).get("result", [])
    if not res:
        return None
    return res[0].get("title", ""), int(res[0].get("pubYear") or 0)


def by_doi(doi: str) -> tuple[str, int] | None:
    d = _get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    if not d:
        return None
    it = d["message"]
    y = (it.get("issued", {}).get("date-parts", [[0]])[0][0]
         or it.get("created", {}).get("date-parts", [[0]])[0][0] or 0)
    return " ".join(it.get("title", []) or []), int(y)


def overlap(a: str, b: str) -> float:
    n = lambda s: set(re.sub(r"\W+", " ", s.lower()).split())    # noqa: E731
    x, y = n(a), n(b)
    return len(x & y) / max(len(x), 1)


def by_title(title: str) -> tuple[str, int] | None:
    """按标题检索 —— 出版商 URL 里推不出 DOI 时的兜底。"""
    d = _get("https://api.crossref.org/works?"
             + urllib.parse.urlencode({"query.bibliographic": title, "rows": 5}))
    if not d:
        return None
    for it in d["message"]["items"]:
        got = " ".join(it.get("title", []) or [])
        if overlap(title, got) >= 0.7:
            y = (it.get("issued", {}).get("date-parts", [[0]])[0][0] or 0)
            return got, int(y)
    return None


def doi_from_url(url: str) -> str | None:
    """从出版商 URL 还原 DOI。

    直接用 `10\\.\\d+/...` 正则会把 Frontiers 链接末尾的 `/full` 一并吞进去，
    DOI 查询随即失败 —— 那是脚本的假阴性，不是文献不存在。
    """
    m = re.search(r"(10\.\d{4,9}/[^\s\"<>]+)", url)
    if m:
        doi = m.group(1)
        # 去掉出版商附加的路径段
        for tail in ("/full", "/abstract", "/pdf", "/html", "/meta"):
            if doi.endswith(tail):
                doi = doi[: -len(tail)]
        return doi.rstrip("/.")
    # Nature 文章号即 DOI 后缀
    m = re.search(r"nature\.com/articles/([a-z0-9\-]+)", url)
    if m:
        return f"10.1038/{m.group(1)}"
    # OUP：/<journal>/article/<vol>/<issue>/<article-id>/...
    m = re.search(r"academic\.oup\.com/(\w+)/article/[^/]+/[^/]+/([a-z0-9]+)", url)
    if m:
        return f"10.1093/{m.group(1)}/{m.group(2)}"
    return None


def main() -> None:
    if TABLE is None or not TABLE.exists():
        # 示例里不写具体盘符路径：那既是假的，又会让发布前的泄露扫描命中一条
        # 不存在的问题。用占位符说明要填什么即可。
        print(f"跳过：未指定参数对照表。设置 {TABLE_ENV} 指向该 CSV 后再跑：\n"
              f'    $env:{TABLE_ENV} = "<该 CSV 的完整路径>"\n'
              "该表是仓库外的私人笔记，不随本仓库分发。")
        raise SystemExit(0)
    rows = list(csv.DictReader(TABLE.open(encoding="utf-8-sig")))
    audio = [r for r in rows
             if any(t in r["category"].lower() for t in AUDIO_TAGS)]
    print(f"表内 {len(rows)} 条，其中声学相关 {len(audio)} 条\n")
    print("=" * 82)

    ok = warn = bad = 0
    for r in audio:
        url = r["source_url"].strip()
        # 表里记的书目标题形如 "Author et al., Title"，取逗号后的部分比对
        claimed = r["literature"].split(",", 1)[-1].strip()
        try:
            claimed_year = int(r["year"])
        except ValueError:
            claimed_year = 0

        got, how = None, ""
        m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url)
        if m:
            got, how = by_pmid(m.group(1)), "PMID"
        if got is None:
            doi = doi_from_url(url)
            if doi:
                got, how = by_doi(doi), "DOI"
        if got is None:                       # 出版商 URL 推不出 DOI 时兜底
            got, how = by_title(claimed), "标题检索"

        if got is None:
            bad += 1
            print(f"\n❌ 解析不到  {r['literature'][:62]}")
            print(f"   {url}")
            continue

        title, year = got
        ov = overlap(claimed, title)
        yr_ok = claimed_year == 0 or abs(year - claimed_year) <= 1
        if ov >= 0.6 and yr_ok:
            ok += 1
            mark = "✅"
        else:
            warn += 1
            mark = "⚠️"
        print(f"\n{mark} {r['literature'][:62]}   [{how}]")
        if mark == "⚠️":
            print(f"   表中：{claimed[:70]}  ({claimed_year})")
            print(f"   实为：{title[:70]}  ({year})")
            print(f"   标题重合 {ov:.2f}")
        time.sleep(0.4)

    print("\n" + "=" * 82)
    print(f"✅ 对上 {ok}    ⚠️ 有出入 {warn}    ❌ 解析不到 {bad}")
    print("\n⚠️ 本脚本只核「文献存在且是这一篇」。")
    print("   表中对结论的概括是否忠实于原文，只能读原文判断 ——")
    print("   核不出问题不等于结论正确。")


if __name__ == "__main__":
    main()
