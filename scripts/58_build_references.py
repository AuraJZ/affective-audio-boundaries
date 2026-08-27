"""逐条核实参考文献并生成 references.bib —— 不核实的不引。

    .venv/Scripts/python scripts/58_build_references.py

## 纪律

`literature-writer` 要求引文只能来自可核验的数据库。本项目此前没有 `references/`，
因此这里从零建一个：**每一条都用 Crossref / Europe PMC 查一次**，
拿到 DOI 与完整元数据才写进 bib；查不到的**列入未核实清单，正文不引**。

凭记忆写参考文献是幻觉最集中的地方 —— 作者、年份、卷期页码单独看都合理，
合起来却指向一篇不存在的文章。**唯一的防线是逐条查。**

## 输出

- `references/references.bib` —— 核实过的条目
- `references/index.yaml` —— 元数据与用途
- `references/UNVERIFIED.md` —— 查不到的，附检索词供人工处理
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request

from soundml.provenance import REPO_ROOT

OUT = REPO_ROOT / "references"
UA = "soundml-refs/1.0 (mailto:research@example.org)"

# (引用键, 期望标题, 用途, DOI 或 None)
#
# **DOI 优先，标题反向核验**：由我给出 DOI，Crossref 返回真实标题，
# 标题与期望不符即拒绝。这个方向能抓住「记错 DOI」，而按标题检索
# 只能抓住「查不到」—— 抓不住「查到了别的」。首版就因此混入了
# Tripathi 2022、Douville 2026 等五篇不相干文献。
WANTED = [
    ("aljanaki2017", "Developing a benchmark for emotional analysis of music", "DEAM 语料", "10.1371/journal.pone.0173392"),
    ("zhang2018pmemo", "The PMEmo Dataset for Music Emotion Recognition", "PMEmo 语料", "10.1145/3206025.3206037"),
    ("fan2017emo", "Emo-soundscapes: A dataset for soundscape emotion recognition", "环境声语料", "10.1109/acii.2017.8273600"),
    ("eerola2011", "A comparison of the discrete and dimensional models of emotion in music", "Soundtracks 语料", "10.1177/0305735610362821"),
    ("daly2018data", "Neural and physiological data from participants listening to affective music", "ds002721 数据集", "10.1038/s41597-020-0507-6"),
    ("daly2014", "Neural correlates of emotional responses to music: an EEG study", "同数据集的原分析", "10.1016/j.neulet.2014.05.003"),
    ("kutt2022", "BIRAFFE2, a multimodal dataset for emotion-based personalization", "BIRAFFE2", "10.1038/s41597-022-01402-6"),
    ("piczak2015", "ESC", "ESC-50", "10.1145/2733373.2806390"),
    ("breiman2001", "Statistical Modeling: The Two Cultures", "Rashomon 集合", "10.1214/ss/1009213726"),
    ("benjamini1995", "Controlling the False Discovery Rate", "BH-FDR", "10.1111/j.2517-6161.1995.tb02031.x"),
    ("maris2007", "Nonparametric statistical testing of EEG- and MEG-data", "符号翻转置换", "10.1016/j.jneumeth.2007.03.024"),
    ("krumhansl1982", "Tracing the dynamic changes in perceived tonal organization", "KK 调性剖面", "10.1037/0033-295x.89.4.334"),
    ("plomp1965", "Tonal consonance and critical bandwidth", "感觉粗糙度", "10.1121/1.1909741"),
    ("mcfee2015", "librosa: Audio and Music Signal Analysis in Python", "特征提取", "10.25080/majora-7b98e3ed-003"),
    ("gramfort2013", "MEG and EEG data analysis with MNE-Python", "脑电处理", "10.3389/fnins.2013.00267"),
    ("makowski2021", "NeuroKit2", "EDA 处理", "10.3758/s13428-020-01516-y"),
    ("ledoit2004", "A well-conditioned estimator for large-dimensional covariance matrices", "收缩协方差", "10.1016/s0047-259x(03)00096-4"),
    ("chen2016xgboost", "XGBoost", "梯度提升", "10.1145/2939672.2939785"),
    ("russell1980", "A circumplex model of affect", "情绪环状模型", "10.1037/h0077714"),
    ("eerola2013", "A Review of Music and Emotion Studies", "音乐情绪综述", "10.1525/mp.2012.30.3.307"),
    ("papalambros2017", "Acoustic Enhancement of Sleep Slow Oscillations and Concomitant Memory Improvement in Older Adults", "声学干预睡眠", "10.3389/fnhum.2017.00109"),
    ("jespersen2022", "Music for insomnia in adults", "音乐助眠 Cochrane", "10.1002/14651858.CD010459.pub3"),
    ("elizalde2023", "Clap Learning Audio Concepts from Natural Language Supervision", "音频-文本嵌入", "10.1109/ICASSP49357.2023.10095889"),
    ("gorgolewski2016", "The brain imaging data structure", "BIDS", "10.1038/sdata.2016.44"),
    ("pernet2019", "EEG-BIDS", "EEG-BIDS", "10.1038/s41597-019-0104-8"),
    ("koelsch2014", "Brain correlates of music-evoked emotions", "音乐情绪神经基础", "10.1038/nrn3666"),
    ("boucsein2012", "Publication recommendations for electrodermal measurements", "EDA 规范", "10.1111/j.1469-8986.2012.01384.x"),
]


MIN_OVERLAP = 0.85       # 首版取 0.6，放进来五篇不相干文献


def _overlap(want: str, got: str) -> float:
    n = lambda s: set(re.sub(r"\W+", " ", s.lower()).split())     # noqa: E731
    a, b = n(want), n(got)
    if not a:
        return 0.0
    # 双向覆盖：既要期望标题被覆盖，也要返回标题不过分冗长
    return min(len(a & b) / len(a), len(a & b) / max(len(b), 1) * 1.6)


def _get(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            return json.load(r)
    except Exception:                                        # noqa: BLE001
        return None


def crossref(title: str, doi: str | None) -> tuple[dict | None, str]:
    """DOI 优先并**反向核验标题**；无 DOI 时按标题检索，阈值从严。

    方向很关键：由我给出 DOI、让 Crossref 返回真实标题再比对，
    能抓住「记错 DOI」；而按标题检索只能抓住「查不到」，
    抓不住「查到了别的」。
    """
    if doi:
        d = _get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
        if d is None:
            return None, "DOI 查询失败"
        it = d["message"]
        got = " ".join(it.get("title", []) or [])
        # 单向：期望标题的词被返回标题覆盖即可。期望标题是我刻意写短的，
        # 双向比较会把 "Controlling the False Discovery Rate" 与其完整标题
        # 判为不符 —— 那是正确文献。
        n = lambda x: set(re.sub(r"\W+", " ", x.lower()).split())
        a, b = n(title), n(got)
        if not a or len(a & b) / len(a) < MIN_OVERLAP:
            return None, f"DOI 指向的是《{got[:60]}》，与期望不符"
        return it, "DOI 反向核验通过"

    q = urllib.parse.urlencode({"query.bibliographic": title, "rows": 5})
    d = _get(f"https://api.crossref.org/works?{q}")
    if d is None:
        return None, "检索失败"
    for it in d["message"]["items"]:
        got = " ".join(it.get("title", []) or [])
        if _overlap(title, got) >= MIN_OVERLAP:
            return it, "标题检索匹配"
    return None, "标题检索无充分匹配"


def bibtex(key: str, it: dict) -> str:
    authors = " and ".join(
        f"{a.get('family','')}, {a.get('given','')}".strip(", ")
        for a in it.get("author", [])[:15]) or "Unknown"
    year = (it.get("issued", {}).get("date-parts", [[None]])[0][0]
            or it.get("created", {}).get("date-parts", [[None]])[0][0])
    title = " ".join(it.get("title", []) or [])
    venue = " ".join(it.get("container-title", []) or [])
    typ = {"journal-article": "article", "proceedings-article": "inproceedings",
           "book-chapter": "incollection", "posted-content": "misc"}.get(
        it.get("type", ""), "misc")
    fields = [f"  author = {{{authors}}}", f"  title = {{{title}}}",
              f"  year = {{{year}}}"]
    if venue:
        fields.append(f"  {'booktitle' if typ=='inproceedings' else 'journal'} = "
                      f"{{{venue}}}")
    for k, f in (("volume", "volume"), ("issue", "number"), ("page", "pages")):
        if it.get(k):
            fields.append(f"  {f} = {{{it[k]}}}")
    if it.get("DOI"):
        fields.append(f"  doi = {{{it['DOI']}}}")
    return f"@{typ}{{{key},\n" + ",\n".join(fields) + "\n}\n"


def main() -> None:
    # 🔴 references.bib 现在是**手工维护**的，重跑本脚本会把它整个覆盖掉。
    #
    # Crossref 的投递有三类系统性缺陷，都已在 bib 里逐条修好，而且**无法**从
    # Crossref 重新取回：ACM 把冒号后的副标题截掉（"ESC"、"XGBoost"）、作者数组
    # 在 15 人处截断且不加 "and others"、部分投递给的是网络首发年而非期号年。
    # 另有 5 条（ARAUS、LAION-CLAP、ds003690 与两条 Ribeiro）根本不在 WANTED 里。
    #
    # 无脑覆盖会把这些全部抹掉，而且不会有任何提示 —— 下一次编译出来的参考文献
    # 表会重新变成 "Esc." 和 "Unknown."。所以先看有没有手工维护标记。
    bib = OUT / "references.bib"
    if bib.exists() and "HAND-MAINTAINED" in bib.read_text(encoding="utf-8"):
        if "--force" not in sys.argv:
            print("🔴 references/references.bib 带有手工维护标记，已停止。\n")
            print("   里面有若干 Crossref 取不回来的修正（截断的标题、截断的作者表、")
            print("   网络首发年），以及 5 条不在 WANTED 列表里的条目。")
            print("   覆盖它们会让参考文献表退回到 \"Esc.\" 和 \"Unknown.\"。\n")
            print("   确实要重新生成，请先把 WANTED 补全，再加 --force。")
            raise SystemExit(1)
        print("⚠️ --force：正在覆盖手工维护的 references.bib\n")

    OUT.mkdir(exist_ok=True)
    ok, bad, index = [], [], []
    for key, title, use, doi in WANTED:
        it, why = crossref(title, doi)
        if it is None:
            bad.append((key, title, use, why))
            print(f"  ❌ {key:18s} {why}")
        else:
            ok.append(bibtex(key, it))
            y = (it.get("issued", {}).get("date-parts", [[None]])[0][0])
            first = (it.get("author", [{}])[0].get("family", "?")
                     if it.get("author") else "?")
            index.append({"key": key, "first_author": first, "year": y,
                          "title": " ".join(it.get("title", []) or []),
                          "doi": it.get("DOI"), "used_for": use})
            print(f"  ✅ {key:18s} {first} {y}  doi:{it.get('DOI')}")
        time.sleep(0.4)                                      # 礼貌限速

    (OUT / "references.bib").write_text("\n".join(ok), encoding="utf-8")
    (OUT / "index.yaml").write_text(
        "\n".join(f"- key: {e['key']}\n  first_author: {e['first_author']}\n"
                  f"  year: {e['year']}\n  doi: {e['doi']}\n"
                  f"  title: \"{e['title']}\"\n  used_for: \"{e['used_for']}\""
                  for e in index), encoding="utf-8")
    # 🔴 `bad` 为空时必须**删掉**已有的 UNVERIFIED.md，不能只是不写。
    #
    # 原来只有 `if bad:` 分支。于是某次运行留下的文件在下一次全部核实通过后
    # 依然躺在仓库里，而且没有任何东西会再碰它。它就那样活了下来，宣称
    # zhang2018pmemo / kutt2022 / piczak2015「正文不得引用」——
    # 这三条全部核实无误，且全部被正文引用（PMEmo、BIRAFFE2、ESC-50 三个语料
    # 的出处），与手稿正面冲突。一份公开的仓库里带着这种文件，读者只能认为
    # 作者引用了自己标为不可信的文献。
    #
    # 报告失败的文件，其生命周期必须绑定在失败本身上。
    unverified = OUT / "UNVERIFIED.md"
    if bad:
        unverified.write_text(
            "# 未能核实的条目 —— **正文不得引用**\n\n"
            "凭记忆补写参考文献是幻觉最集中的地方。以下条目未通过核验，\n"
            "**正文一律不引**，需人工检索确认后再加入。\n\n"
            "多数是会议论文集（NeurIPS / JMLR / SciPy）不在 Crossref，\n"
            "并非文献不存在 —— 但既然核验不了，就不引。\n\n"
            + "\n".join(f"- `{k}` —— {t}\n  - 用途：{u}\n  - 原因：{w}"
                        for k, t, u, w in bad),
            encoding="utf-8")
    elif unverified.exists():
        unverified.unlink()
        print("已删除过期的 references/UNVERIFIED.md —— 本次全部核实通过")

    print(f"\n核实 {len(ok)}/{len(WANTED)}   → references/references.bib")
    if bad:
        print(f"未核实 {len(bad)} 条，已写入 references/UNVERIFIED.md，正文不引")


if __name__ == "__main__":
    main()
