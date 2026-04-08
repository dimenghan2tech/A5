from __future__ import annotations

import base64
import html
import json
import os
import re
from pathlib import Path
from typing import Any

import requests
import streamlit as st

st.set_page_config(
    page_title="篇章分析综合平台",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

import spacy
from spacy.language import Language


BASE_DIR = Path(__file__).resolve().parent
LOCAL_SAMPLE_PATH = BASE_DIR / "data" / "sample_discourse.json"
NEURAL_SAMPLE_CACHE_PATH = BASE_DIR / "data" / "neuraleduseg_sample.json"
HF_MODEL_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "models--biu-nlp--f-coref"
DEFAULT_REMOTE_URL = "https://jsonplaceholder.typicode.com/posts"
NEURAL_EDU_API = "https://api.github.com/repos/PKU-TANGENT/NeuralEDUSeg/contents/data/rst"
NEURAL_EDU_RAW_PREFIX = "https://raw.githubusercontent.com/PKU-TANGENT/NeuralEDUSeg/master/data/rst"
KNOWN_NEURAL_SAMPLE_PAIRS = [
    {
        "name": "wsj_0605",
        "raw_url": f"{NEURAL_EDU_RAW_PREFIX}/TRAINING/wsj_0605.out",
        "edu_url": f"{NEURAL_EDU_RAW_PREFIX}/TRAINING/wsj_0605.out.edus",
        "path": "data/rst/TRAINING/wsj_0605.out.edus",
    },
    {
        "name": "wsj_0601",
        "raw_url": f"{NEURAL_EDU_RAW_PREFIX}/TRAINING/wsj_0601.out",
        "edu_url": f"{NEURAL_EDU_RAW_PREFIX}/TRAINING/wsj_0601.out.edus",
        "path": "data/rst/TRAINING/wsj_0601.out.edus",
    },
]

CONNECTIVE_MAP = {
    "because": ("Contingency", "Cause"),
    "so": ("Contingency", "Result"),
    "therefore": ("Contingency", "Result"),
    "thus": ("Contingency", "Result"),
    "if": ("Contingency", "Condition"),
    "when": ("Temporal", "Asynchronous"),
    "while": ("Temporal", "Synchronous"),
    "before": ("Temporal", "Precedence"),
    "after": ("Temporal", "Succession"),
    "but": ("Comparison", "Contrast"),
    "however": ("Comparison", "Contrast"),
    "although": ("Comparison", "Concession"),
    "though": ("Comparison", "Concession"),
    "and": ("Expansion", "Conjunction"),
    "also": ("Expansion", "Conjunction"),
    "instead": ("Expansion", "Alternative"),
    "for example": ("Expansion", "Instantiation"),
    "in addition": ("Expansion", "Conjunction"),
}

EXPLICIT_CONNECTIVES = {
    "when": "Temporal",
    "after": "Temporal",
    "before": "Temporal",
    "while": "Temporal",
    "since": "Ambiguous",
    "because": "Contingency",
    "therefore": "Contingency",
    "thus": "Contingency",
    "if": "Contingency",
    "but": "Comparison",
    "although": "Comparison",
    "though": "Comparison",
    "however": "Comparison",
    "and": "Expansion",
    "or": "Expansion",
    "also": "Expansion",
}

PRONOUNS = {
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "they",
    "them",
    "their",
    "theirs",
    "it",
    "its",
}

SUBORDINATORS = {
    "because",
    "although",
    "though",
    "when",
    "while",
    "if",
    "before",
    "after",
    "since",
    "unless",
    "whereas",
}

BOUNDARY_DEP_LABELS = {"advcl", "ccomp", "xcomp", "relcl", "acl"}
BOUNDARY_MARKERS = {";", ":", ","}

FALLBACK_NEURAL_SAMPLE = {
    "path": "data/neuraleduseg_sample.json",
    "download_url": "local-cache",
    "source_note": "项目内缓存的 NeuralEDUSeg 公开样本",
}


@st.cache_resource(show_spinner=False)
def load_spacy_model() -> Language:
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp


@st.cache_resource(show_spinner=False)
def load_fastcoref_model() -> tuple[Any, str]:
    try:
        from fastcoref import FCoref
        bad_proxy_values = {"http://127.0.0.1:9", "https://127.0.0.1:9"}
        cleared = []
        for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "GIT_HTTP_PROXY", "GIT_HTTPS_PROXY"]:
            value = os.environ.get(key, "")
            if value in bad_proxy_values or value.endswith("127.0.0.1:9"):
                os.environ.pop(key, None)
                cleared.append(key)

        snapshots_dir = HF_MODEL_CACHE_DIR / "snapshots"
        snapshot_candidates = sorted(
            [path for path in snapshots_dir.iterdir() if path.is_dir()],
            reverse=True,
        ) if snapshots_dir.exists() else []

        if snapshot_candidates:
            local_model_path = str(snapshot_candidates[0])
            model = FCoref(model_name_or_path=local_model_path, device="cpu")
            note = ""
            if cleared:
                note = f"已自动清理代理变量：{', '.join(cleared)}。"
            return model, note

        model = FCoref(device="cpu")
        note = ""
        if cleared:
            note = f"已自动清理代理变量：{', '.join(cleared)}。"
        return model, note
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def load_local_samples() -> list[dict[str, str]]:
    with LOCAL_SAMPLE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_cached_neural_sample() -> dict[str, Any]:
    with NEURAL_SAMPLE_CACHE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #102a43;
            --muted: #52606d;
            --line: rgba(15, 23, 42, 0.08);
            --panel: rgba(255, 255, 255, 0.9);
            --blue: #1877f2;
            --green: #00a676;
            --shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
        }
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(24, 119, 242, 0.14), transparent 30%),
                radial-gradient(circle at left bottom, rgba(0, 166, 118, 0.11), transparent 28%),
                linear-gradient(180deg, #f8fbff 0%, #edf3f8 45%, #eef2f6 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        .hero {
            padding: 1.5rem 1.7rem;
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.97), rgba(255,255,255,0.84)),
                radial-gradient(circle at top right, rgba(24,119,242,0.16), transparent 36%);
            border: 1px solid var(--line);
            box-shadow: var(--shadow);
            margin-bottom: 1.15rem;
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
        }
        .hero::after {
            content: "";
            position: absolute;
            right: -30px;
            top: -36px;
            width: 180px;
            height: 180px;
            background: radial-gradient(circle, rgba(24,119,242,0.18), transparent 64%);
            pointer-events: none;
        }
        .hero-kicker {
            display: inline-block;
            padding: 0.28rem 0.72rem;
            margin-bottom: 0.75rem;
            border-radius: 999px;
            background: rgba(16, 42, 67, 0.06);
            color: #0f4c81;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.05em;
        }
        .hero h1 {
            margin: 0 0 0.4rem 0;
            color: var(--ink);
            font-size: 2.15rem;
            letter-spacing: -0.03em;
        }
        .hero p {
            margin: 0;
            color: #486581;
            line-height: 1.8;
            max-width: 820px;
        }
        .metric-card {
            padding: 1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid var(--line);
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            min-height: 132px;
        }
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 18px 32px rgba(15, 23, 42, 0.10);
        }
        .metric-label {
            color: #486581;
            font-size: 0.92rem;
        }
        .metric-value {
            font-size: 1.7rem;
            font-weight: 700;
            color: var(--ink);
            margin: 0.35rem 0;
        }
        .metric-caption {
            color: #7b8794;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .tag {
            display: inline-block;
            padding: 0.2rem 0.58rem;
            margin-right: 0.4rem;
            border-radius: 999px;
            background: #e4f7ef;
            color: #116149;
            font-size: 0.82rem;
            border: 1px solid rgba(0, 166, 118, 0.12);
        }
        .edu-box {
            padding: 1rem 1.05rem;
            border-left: 4px solid #1877f2;
            background: rgba(255,255,255,0.94);
            border-radius: 16px;
            margin-bottom: 0.75rem;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.05);
            border: 1px solid var(--line);
        }
        .edu-box.pred {
            border-left-color: #1877f2;
        }
        .edu-box.gold {
            border-left-color: #00a676;
        }
        .token {
            display: inline-block;
            padding: 0.1rem 0.28rem;
            margin: 0 0.1rem 0.18rem 0;
            border-radius: 8px;
        }
        .token.boundary {
            background: #ffe3e3;
            color: #9b2226;
            font-weight: 700;
        }
        .token.gold-boundary {
            background: #dff7ec;
            color: #116149;
            font-weight: 700;
        }
        .compare-panel {
            padding: 1.05rem 1.15rem;
            border-radius: 18px;
            background: rgba(255,255,255,0.9);
            border: 1px solid var(--line);
            margin-bottom: 1rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.05);
        }
        .note-box {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(16, 42, 67, 0.045), rgba(16, 42, 67, 0.025));
            border: 1px solid rgba(16, 42, 67, 0.08);
            color: #243b53;
        }
        .sense-chip {
            display: inline-block;
            padding: 0.18rem 0.6rem;
            border-radius: 999px;
            margin-left: 0.35rem;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .sense-temporal {
            background: #e0f2fe;
            color: #075985;
        }
        .sense-contingency {
            background: #fee2e2;
            color: #991b1b;
        }
        .sense-comparison {
            background: #ede9fe;
            color: #5b21b6;
        }
        .sense-expansion {
            background: #dcfce7;
            color: #166534;
        }
        .sense-ambiguous {
            background: #fef3c7;
            color: #92400e;
        }
        .arg-box {
            padding: 1.05rem 1.1rem;
            border-radius: 18px;
            min-height: 140px;
            border: 1px solid var(--line);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }
        .arg1-box {
            background: linear-gradient(180deg, rgba(24, 119, 242, 0.1), rgba(24, 119, 242, 0.05));
        }
        .arg2-box {
            background: linear-gradient(180deg, rgba(0, 166, 118, 0.12), rgba(0, 166, 118, 0.06));
        }
        .inline-conn {
            font-weight: 800;
            padding: 0.05rem 0.35rem;
            border-radius: 10px;
            margin: 0 0.1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.55rem;
            background: rgba(255,255,255,0.62);
            padding: 0.45rem;
            border: 1px solid var(--line);
            border-radius: 18px;
            margin-bottom: 1rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 48px;
            background: transparent;
            border-radius: 14px;
            padding: 0 1rem;
            color: var(--muted);
            font-weight: 700;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(24,119,242,0.12), rgba(0,166,118,0.08));
            color: var(--ink);
        }
        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }
        .stTextArea textarea {
            border-radius: 16px !important;
            border: 1px solid var(--line) !important;
            background: rgba(255,255,255,0.95) !important;
        }
        .stSelectbox > div > div,
        .stMultiSelect > div > div {
            border-radius: 14px !important;
        }
        .stDataFrame, div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid var(--line);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
            background: rgba(255,255,255,0.95);
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 18px;
            background: rgba(255,255,255,0.86);
            box-shadow: 0 8px 20px rgba(15,23,42,0.04);
        }
        h3 {
            color: var(--ink);
            letter-spacing: -0.02em;
        }
        @media (max-width: 760px) {
            .hero {
                padding: 1.2rem 1rem;
            }
            .hero h1 {
                font-size: 1.75rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="metric-value">{html.escape(value)}</div>
            <div class="metric-caption">{html.escape(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def split_clause_on_markers(sentence: str) -> list[str]:
    markers = [
        "because",
        "but",
        "although",
        "however",
        "when",
        "while",
        "if",
        "so",
        "therefore",
        "and",
    ]
    pattern = r"\s+(?=(?:%s)\b)" % "|".join(re.escape(marker) for marker in markers)
    parts = re.split(pattern, sentence, flags=re.IGNORECASE)
    return [part.strip(" ,;") for part in parts if part.strip(" ,;")] or [sentence.strip()]


def fetch_remote_samples(limit: int = 5) -> tuple[list[dict[str, str]], str]:
    try:
        response = requests.get(DEFAULT_REMOTE_URL, timeout=8)
        response.raise_for_status()
        rows = response.json()[:limit]
        samples = []
        for row in rows:
            samples.append(
                {
                    "title": row["title"].strip().capitalize(),
                    "text": row["body"].replace("\n", " ").strip(),
                    "source": DEFAULT_REMOTE_URL,
                }
            )
        return samples, "已从网络获取实时文本样本。"
    except Exception:
        return load_local_samples(), "网络样本获取失败，已自动切换到本地样例。"


def flatten_github_tree(api_url: str, depth: int = 2) -> list[dict[str, Any]]:
    if depth < 0:
        return []
    response = requests.get(api_url, timeout=12, headers={"Accept": "application/vnd.github+json"})
    response.raise_for_status()
    items = response.json()
    if isinstance(items, dict):
        items = [items]

    files: list[dict[str, Any]] = []
    for item in items:
        if item.get("type") == "file":
            files.append(item)
        elif item.get("type") == "dir" and depth > 0:
            files.extend(flatten_github_tree(item["url"], depth - 1))
    return files


def guess_is_edu_file(path: str) -> bool:
    lowered = path.lower()
    return any(
        token in lowered
        for token in [".edus", ".edu", ".out", ".txt", ".preprocessed", "sample", "wsj_"]
    )


def decode_github_content(item: dict[str, Any]) -> str:
    if item.get("download_url"):
        response = requests.get(item["download_url"], timeout=12)
        response.raise_for_status()
        return response.text
    if item.get("content"):
        return base64.b64decode(item["content"]).decode("utf-8", errors="ignore")
    raise ValueError("No downloadable content found.")


def parse_token_label_format(raw_text: str) -> dict[str, Any] | None:
    tokens: list[str] = []
    boundary_indices: list[int] = []
    gold_edus: list[str] = []
    current_tokens: list[str] = []

    valid_rows = 0
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current_tokens:
                gold_edus.append(" ".join(current_tokens))
                current_tokens = []
            continue
        parts = re.split(r"\s+", stripped)
        if len(parts) < 2:
            continue
        token = parts[0]
        label = parts[-1].lower()
        if not re.search(r"(0|1|b|e|edu)", label):
            continue
        valid_rows += 1
        tokens.append(token)
        current_tokens.append(token)
        if label in {"1", "b", "b-edu", "e", "e-edu", "edu_break", "edu"}:
            boundary_indices.append(len(tokens) - 1)
            gold_edus.append(" ".join(current_tokens))
            current_tokens = []

    if current_tokens:
        gold_edus.append(" ".join(current_tokens))

    if valid_rows >= 5 and gold_edus:
        return {
            "raw_text": " ".join(tokens),
            "edus": gold_edus,
            "boundary_indices": boundary_indices,
            "format": "token-label",
        }
    return None


def parse_edu_line_format(raw_text: str) -> dict[str, Any] | None:
    candidates = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("//"):
            continue
        stripped = re.sub(r"^<EDU[^>]*>", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"</EDU>$", "", stripped, flags=re.IGNORECASE)
        stripped = stripped.strip()
        if len(stripped.split()) >= 2:
            candidates.append(stripped)

    if len(candidates) >= 2:
        return {
            "raw_text": " ".join(candidates),
            "edus": candidates,
            "boundary_indices": compute_boundary_indices(" ".join(candidates), candidates),
            "format": "edu-per-line",
        }
    return None


def parse_inline_marker_format(raw_text: str) -> dict[str, Any] | None:
    marker_patterns = [r"<EDU>", r"</EDU>", r"\|\|\|", r"<seg>", r"</seg>"]
    if not any(re.search(pattern, raw_text, flags=re.IGNORECASE) for pattern in marker_patterns):
        return None

    text = re.sub(r"</?(EDU|seg)[^>]*>", "|||", raw_text, flags=re.IGNORECASE)
    text = text.replace("\n", " ")
    segments = [seg.strip(" |") for seg in text.split("|||") if seg.strip(" |")]
    if len(segments) >= 2:
        merged_text = " ".join(segments)
        return {
            "raw_text": merged_text,
            "edus": segments,
            "boundary_indices": compute_boundary_indices(merged_text, segments),
            "format": "inline-marker",
        }
    return None


def compute_boundary_indices(raw_text: str, edus: list[str]) -> list[int]:
    raw_tokens = raw_text.split()
    boundary_indices: list[int] = []
    cursor = 0
    for edu in edus:
        edu_tokens = edu.split()
        if not edu_tokens:
            continue
        cursor += len(edu_tokens)
        boundary_indices.append(min(cursor - 1, max(len(raw_tokens) - 1, 0)))
    return sorted({idx for idx in boundary_indices if idx >= 0})


def parse_edus_file(raw_text: str, edu_text: str) -> dict[str, Any]:
    edus = [line.strip() for line in edu_text.splitlines() if line.strip()]
    merged_raw = normalize_text(raw_text.replace("\n", " "))
    return {
        "raw_text": merged_raw,
        "edus": edus,
        "boundary_indices": compute_boundary_indices(merged_raw, edus),
        "format": "paired .out + .out.edus",
    }


def load_neural_edu_sample() -> dict[str, Any]:
    diagnostics: list[str] = []

    try:
        requests.get(NEURAL_EDU_API, timeout=10).raise_for_status()
        diagnostics.append("GitHub Contents API 可访问。")
    except Exception as exc:
        diagnostics.append(f"GitHub Contents API 不可访问：{exc}")

    for pair in KNOWN_NEURAL_SAMPLE_PAIRS:
        try:
            raw_response = requests.get(pair["raw_url"], timeout=10)
            raw_response.raise_for_status()
            edu_response = requests.get(pair["edu_url"], timeout=10)
            edu_response.raise_for_status()
            parsed = parse_edus_file(raw_response.text, edu_response.text)
            parsed.update(
                {
                    "path": pair["path"],
                    "download_url": pair["edu_url"],
                    "source_note": "来自 NeuralEDUSeg 仓库公开样本",
                    "diagnostics": diagnostics
                    + [f"成功抓取公开样本对：{pair['name']} (.out + .out.edus)。"],
                    "failure_reason": "",
                }
            )
            return parsed
        except Exception as exc:
            diagnostics.append(f"{pair['name']} 抓取失败：{exc}")

    fallback = dict(FALLBACK_NEURAL_SAMPLE)
    fallback.update(load_cached_neural_sample())
    fallback["boundary_indices"] = compute_boundary_indices(fallback["raw_text"], fallback["edus"])
    fallback["format"] = "cached-public-sample"
    fallback["diagnostics"] = diagnostics + [
        "原因 1：`raw.githubusercontent.com/.../data/rst/` 不能列目录，不能靠 raw 根目录自动发现文件。",
        "原因 2：当前运行环境对 GitHub 请求可能受代理、超时或权限影响，因此在线抓取会失败。",
        "解决方式：优先抓取已知公开样本对，失败后回退到项目内缓存的同源公开样本。",
    ]
    fallback["failure_reason"] = "在线抓取公开样本失败，已切换到项目内缓存样本。"
    return fallback


def edu_boundary_token_indices(edus: list[str]) -> list[int]:
    boundary_indices: list[int] = []
    cursor = 0
    for edu in edus:
        tokens = edu.split()
        if not tokens:
            continue
        cursor += len(tokens)
        boundary_indices.append(cursor - 1)
    return boundary_indices


def explain_rule_boundary(token: Any, next_token: Any, is_last_token: bool) -> str:
    if token.text in {".", "!", "?"}:
        return "句末标点触发切分。"
    if token.text in BOUNDARY_MARKERS and next_token is not None:
        return f"遇到标点 `{token.text}`，规则认为这里可能结束一个从句或短语。"
    if token.dep_ in BOUNDARY_DEP_LABELS and token.dep_:
        return f"依存关系 `{token.dep_}` 被视为潜在从句边界。"
    if token.pos_ == "SCONJ":
        return f"从属连词 `{token.text}` 所在位置提示句法边界。"
    if token.lower_ in SUBORDINATORS:
        return f"连接词 `{token.text}` 命中了从属连词规则。"
    if is_last_token:
        return "句子结束，执行收尾切分。"
    return "规则未命中特殊边界，按句末收尾。"


def segment_discourse_rule_based(text: str, nlp: Language) -> dict[str, Any]:
    doc = nlp(text)
    edus: list[str] = []
    boundaries: list[int] = []
    boundary_reasons: list[str] = []
    current_tokens: list[str] = []
    global_index = -1

    for sent in doc.sents:
        sent_doc = nlp(sent.text)
        for i, token in enumerate(sent_doc):
            token_text = token.text.strip()
            if not token_text:
                continue
            current_tokens.append(token_text)
            global_index += 1

            boundary_now = False
            next_token = sent_doc[i + 1] if i + 1 < len(sent_doc) else None

            if token.text in {".", "!", "?"}:
                boundary_now = True
            elif token.text in BOUNDARY_MARKERS and next_token is not None:
                boundary_now = True
            elif token.dep_ in BOUNDARY_DEP_LABELS and i > 0:
                boundary_now = True
            elif token.pos_ == "SCONJ" and i > 0:
                boundary_now = True
            elif token.lower_ in SUBORDINATORS and i > 0:
                boundary_now = True

            if boundary_now:
                edus.append(" ".join(current_tokens).strip())
                boundaries.append(global_index)
                boundary_reasons.append(
                    explain_rule_boundary(token, next_token, i == len(sent_doc) - 1)
                )
                current_tokens = []

        if current_tokens:
            edus.append(" ".join(current_tokens).strip())
            boundaries.append(global_index)
            boundary_reasons.append("句子扫描结束后，对剩余片段执行收尾切分。")
            current_tokens = []

    clean_edus = [re.sub(r"\s+([,.;:!?])", r"\1", edu) for edu in edus if edu.strip()]
    return {
        "raw_text": text,
        "edus": clean_edus,
        "boundary_indices": edu_boundary_token_indices(clean_edus),
        "boundary_reasons": boundary_reasons[: len(clean_edus)],
    }


def tokens_with_boundary_markup(raw_text: str, boundary_indices: list[int], css_class: str) -> str:
    words = raw_text.split()
    boundary_set = set(boundary_indices)
    spans = []
    for idx, word in enumerate(words):
        token_class = f"token {css_class}" if idx in boundary_set else "token"
        spans.append(f'<span class="{token_class}">{html.escape(word)}</span>')
    return " ".join(spans)


def render_edu_cards(
    edus: list[str],
    boundary_indices: list[int],
    box_class: str,
    boundary_class: str,
    reasons: list[str] | None = None,
) -> None:
    running = 0
    for i, edu in enumerate(edus, start=1):
        tokens = edu.split()
        local_indices = []
        for local_idx in range(len(tokens)):
            global_idx = running + local_idx
            if global_idx in boundary_indices:
                local_indices.append(local_idx)
        running += len(tokens)
        token_html = tokens_with_boundary_markup(edu, local_indices, boundary_class)
        st.markdown(
            f"""
            <div class="edu-box {box_class}">
                <strong>EDU {i}</strong>
                <div style="margin-top: 0.55rem; line-height: 1.9;">{token_html}</div>
                {"<div style='margin-top:0.55rem;color:#52606d;font-size:0.92rem;'><strong>切分原因：</strong>" + html.escape(reasons[i - 1]) + "</div>" if reasons and i - 1 < len(reasons) else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )


def compare_boundaries(predicted: list[int], gold: list[int]) -> dict[str, Any]:
    pred = set(predicted)
    ref = set(gold)
    overlap = pred & ref
    precision = len(overlap) / len(pred) if pred else 0.0
    recall = len(overlap) / len(ref) if ref else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "predicted_count": len(pred),
        "gold_count": len(ref),
        "matched": len(overlap),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_only": sorted(pred - ref),
        "gold_only": sorted(ref - pred),
    }


def explain_gold_boundaries(edus: list[str]) -> list[str]:
    reasons = []
    for idx, edu in enumerate(edus, start=1):
        if idx == len(edus):
            reasons.append("NeuralEDUSeg 公开样本中的最后一个 EDU，按人工标注在此结束。")
        else:
            reasons.append("该边界来自 `.out.edus` 真值标注文件，每一行对应一个人工标注的 EDU。")
    return reasons


def extract_discourse_relations(edus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relations = []
    for left, right in zip(edus, edus[1:]):
        marker = right["marker"]
        if marker == "implicit":
            lowered = right["text"].lower()
            marker = next((m for m in CONNECTIVE_MAP if m in lowered), "implicit")

        if marker in CONNECTIVE_MAP:
            top_level, sense = CONNECTIVE_MAP[marker]
        else:
            top_level, sense = ("Expansion", "Implicit continuation")

        relations.append(
            {
                "from_edu": left["edu_id"],
                "to_edu": right["edu_id"],
                "connective": marker,
                "pdtb_level_1": top_level,
                "pdtb_level_2": sense,
            }
        )
    return relations


def infer_since_sense(sentence: str, match_start: int) -> tuple[str, str]:
    left_context = sentence[:match_start].lower()
    if re.search(r"\b(has|have|had|been|changed|grew|fallen|risen|declined|increased)\b", left_context):
        return "Temporal", "句子左侧更像时间起点或持续变化描述。"
    if re.search(r"\b(result|because|therefore|reason|caused|due)\b", sentence.lower()):
        return "Contingency", "上下文出现因果触发词，倾向解释为原因。"
    if re.search(r"\b\d{4}\b|\byesterday\b|\blast\b|\bthen\b|\bago\b", sentence.lower()):
        return "Temporal", "上下文出现明显时间线索。"
    return "Contingency", "没有明显时间线索时，这里先按因果关系处理。"


def find_explicit_connective(sentence: str) -> dict[str, Any] | None:
    sentence = normalize_text(sentence)
    matches = []
    for connective, coarse_label in EXPLICIT_CONNECTIVES.items():
        pattern = rf"\b{re.escape(connective)}\b"
        found = re.search(pattern, sentence, flags=re.IGNORECASE)
        if found:
            actual_label = coarse_label
            note = ""
            if connective == "since":
                actual_label, note = infer_since_sense(sentence, found.start())
            matches.append(
                {
                    "connective": connective,
                    "display": sentence[found.start() : found.end()],
                    "label": actual_label,
                    "match_start": found.start(),
                    "match_end": found.end(),
                    "note": note,
                }
            )
    if not matches:
        return None
    matches.sort(key=lambda item: item["match_start"])
    return matches[0]


def split_arguments(sentence: str, connective_info: dict[str, Any]) -> tuple[str, str]:
    start = connective_info["match_start"]
    end = connective_info["match_end"]
    arg1 = sentence[:start].strip(" ,;-")
    arg2 = sentence[end:].strip(" ,;-")

    if not arg1:
        parts = re.split(r"[,:;-]\s*", arg2, maxsplit=1)
        if len(parts) == 2:
            arg1 = parts[0].strip()
            arg2 = parts[1].strip()
        else:
            arg1 = connective_info["display"]
    if not arg2:
        arg2 = "(未识别到后置论据)"
    return arg1 or "(未识别到前置论据)", arg2


def sense_class(label: str) -> str:
    return {
        "Temporal": "sense-temporal",
        "Contingency": "sense-contingency",
        "Comparison": "sense-comparison",
        "Expansion": "sense-expansion",
    }.get(label, "sense-ambiguous")


def highlight_connective(sentence: str, info: dict[str, Any]) -> str:
    start = info["match_start"]
    end = info["match_end"]
    label = html.escape(info["label"].upper())
    conn = html.escape(sentence[start:end])
    cls = sense_class(info["label"])
    return (
        html.escape(sentence[:start])
        + f'<span class="inline-conn {cls}">{conn}</span>'
        + f'<span class="sense-chip {cls}">{label}</span>'
        + html.escape(sentence[end:])
    )


def segment_for_relations(text: str, nlp: Language) -> list[dict[str, Any]]:
    doc = nlp(text)
    edus = []
    edu_id = 1
    for sent_id, sent in enumerate(doc.sents, start=1):
        clauses = split_clause_on_markers(sent.text.strip())
        for clause in clauses:
            tokens = clause.split()
            marker = next(
                (token.lower() for token in tokens if token.lower() in CONNECTIVE_MAP),
                None,
            )
            edus.append(
                {
                    "edu_id": edu_id,
                    "sentence_id": sent_id,
                    "text": clause,
                    "token_count": len(tokens),
                    "marker": marker or "implicit",
                }
            )
            edu_id += 1
    return edus


def heuristic_coref(doc: Any) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    mention_lookup: dict[str, int] = {}
    recent_entities: list[str] = []

    noun_phrases: list[str] = []
    try:
        noun_phrases = [chunk.text.strip() for chunk in doc.noun_chunks if chunk.text.strip()]
    except Exception:
        noun_phrases = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", doc.text)

    for phrase in noun_phrases:
        if len(phrase) < 2:
            continue
        lowered = phrase.lower()
        if lowered not in mention_lookup:
            mention_lookup[lowered] = len(clusters)
            clusters.append({"entity": phrase, "mentions": [phrase], "method": "heuristic"})
            recent_entities.append(phrase)

    for token in doc:
        lowered = token.text.lower()
        if lowered in PRONOUNS and recent_entities:
            target = recent_entities[-1]
            idx = mention_lookup[target.lower()]
            clusters[idx]["mentions"].append(token.text)

    return [cluster for cluster in clusters if len(cluster["mentions"]) > 1]


def build_char_spans_from_strings(text: str, mention_strings: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    used_ranges: list[tuple[int, int]] = []
    for mention in mention_strings:
        pattern = re.escape(mention)
        for match in re.finditer(pattern, text):
            span = (match.start(), match.end())
            if span not in used_ranges:
                spans.append(span)
                used_ranges.append(span)
                break
    return spans


def run_coref(text: str, nlp: Language) -> tuple[list[dict[str, Any]], str]:
    model, load_error = load_fastcoref_model()
    if model is not None:
        try:
            predictions = model.predict([text])
            cluster_data = []
            string_clusters = predictions[0].get_clusters(as_strings=True)
            span_clusters = predictions[0].get_clusters(as_strings=False)
            for idx, (cluster_strings, cluster_spans) in enumerate(
                zip(string_clusters, span_clusters),
                start=1,
            ):
                if len(cluster_strings) < 2:
                    continue
                cluster_data.append(
                    {
                        "entity": f"Entity {idx}",
                        "mentions": cluster_strings,
                        "spans": cluster_spans,
                        "method": "fastcoref",
                    }
                )
            if cluster_data:
                return cluster_data, "当前结果来自 fastcoref 神经指代消解模型。"
        except Exception as exc:
            load_error = f"{type(exc).__name__}: {exc}"

    doc = nlp(text)
    heuristic_clusters = heuristic_coref(doc)
    for cluster in heuristic_clusters:
        cluster["spans"] = build_char_spans_from_strings(text, cluster["mentions"])
    detail = f" 具体原因：{load_error}" if load_error else ""
    return heuristic_clusters, f"fastcoref 不可用，当前展示的是启发式回退结果。{detail}"


def render_coref_text(text: str, clusters: list[dict[str, Any]]) -> str:
    palette = [
        "#fde68a",
        "#bfdbfe",
        "#c7f9cc",
        "#fecdd3",
        "#ddd6fe",
        "#a7f3d0",
        "#fdba74",
        "#f9a8d4",
    ]
    span_entries: list[dict[str, Any]] = []
    for cluster_id, cluster in enumerate(clusters, start=1):
        color = palette[(cluster_id - 1) % len(palette)]
        spans = cluster.get("spans", []) or build_char_spans_from_strings(text, cluster.get("mentions", []))
        mention_strings = [m for m in cluster.get("mentions", []) if isinstance(m, str)]
        for mention_id, span in enumerate(spans, start=1):
            if not isinstance(span, (list, tuple)) or len(span) != 2:
                continue
            start, end = int(span[0]), int(span[1])
            if start < 0 or start >= len(text):
                continue
            end_candidates = [end, end + 1, end - 1]
            resolved_end = None
            for candidate in end_candidates:
                if candidate <= start or candidate > len(text):
                    continue
                piece = text[start:candidate].strip()
                if piece and any(piece == mention or piece in mention or mention in piece for mention in mention_strings):
                    resolved_end = candidate
                    break
            if resolved_end is None:
                for mention in sorted(mention_strings, key=len, reverse=True):
                    if text.startswith(mention, start):
                        resolved_end = start + len(mention)
                        break
            if resolved_end is None:
                continue
            end = resolved_end
            if start >= end or end > len(text):
                continue
            span_entries.append(
                {
                    "start": start,
                    "end": end,
                    "cluster_id": cluster_id,
                    "mention_id": mention_id,
                    "color": color,
                }
            )

    span_entries.sort(key=lambda item: (item["start"], -(item["end"] - item["start"])))
    filtered: list[dict[str, Any]] = []
    current_end = -1
    for entry in span_entries:
        if entry["start"] >= current_end:
            filtered.append(entry)
            current_end = entry["end"]

    html_parts = []
    cursor = 0
    for entry in filtered:
        if cursor < entry["start"]:
            html_parts.append(html.escape(text[cursor : entry["start"]]))
        mention_text = html.escape(text[entry["start"] : entry["end"]])
        html_parts.append(
            f'<span style="background:{entry["color"]};padding:0.12rem 0.3rem;border-radius:0.45rem;'
            f'box-shadow: inset 0 0 0 1px rgba(15,23,42,0.08);font-weight:600;" '
            f'title="Cluster {entry["cluster_id"]} / Mention {entry["mention_id"]}">{mention_text}</span>'
        )
        cursor = entry["end"]
    if cursor < len(text):
        html_parts.append(html.escape(text[cursor:]))

    return "".join(html_parts).replace("\n", "<br/>")


def render_module_one(text: str, nlp: Language) -> None:
    st.subheader("模块 1：话语分割（规则基线 vs NeuralEDUSeg 真实数据）")
    st.caption(
        "数据源来自 PKU-TANGENT/NeuralEDUSeg 的公开 `data/rst/` 样本；"
        "仓库 README 说明这里只提供少量样本用于展示数据结构。"
    )

    with st.spinner("正在解析 NeuralEDUSeg 公开样本并运行规则基线..."):
        neural_sample = load_neural_edu_sample()
        baseline = segment_discourse_rule_based(neural_sample["raw_text"], nlp)
        comparison = compare_boundaries(
            baseline["boundary_indices"],
            neural_sample["boundary_indices"],
        )
        gold_reasons = explain_gold_boundaries(neural_sample["edus"])

    st.markdown(
        f"""
        <div class="compare-panel">
            <strong>已抓取样本：</strong> {html.escape(neural_sample["path"])}<br/>
            <strong>解析格式：</strong> {html.escape(neural_sample["format"])}<br/>
            <strong>原始来源：</strong> {html.escape(neural_sample["source_note"])}<br/>
            <strong>抓取状态：</strong> {html.escape(neural_sample.get("failure_reason") or "在线抓取成功")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        render_metric_card("规则边界数", str(comparison["predicted_count"]), "启发式基线预测出的 EDU 边界个数")
    with metric_col2:
        render_metric_card("真实边界数", str(comparison["gold_count"]), "NeuralEDUSeg 样本中的真实边界个数")
    with metric_col3:
        render_metric_card("边界匹配数", str(comparison["matched"]), "两种切分结果在边界词上的重合数")
    with metric_col4:
        render_metric_card("边界 F1", f'{comparison["f1"]:.2f}', "仅用于课堂展示的轻量对比指标")

    st.markdown("### 当前用于对比的原始文本")
    st.write(neural_sample["raw_text"])

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("#### 左栏：规则基线切分结果")
        st.caption("边界词用红色高亮；每个 EDU 下方显示这次切分触发的具体规则。")
        render_edu_cards(
            baseline["edus"],
            baseline["boundary_indices"],
            "pred",
            "boundary",
            baseline.get("boundary_reasons"),
        )
    with right_col:
        st.markdown("#### 右栏：NeuralEDUSeg 数据集真实标注")
        st.caption("真实样本中的 EDU 末词用绿色高亮；切分原因来自 `.out.edus` 真值文件。")
        render_edu_cards(
            neural_sample["edus"],
            neural_sample["boundary_indices"],
            "gold",
            "gold-boundary",
            gold_reasons,
        )

    st.markdown("### 差异观察")
    diff_left, diff_right = st.columns(2)
    with diff_left:
        st.dataframe(
            [
                {
                    "指标": "Precision",
                    "数值": round(comparison["precision"], 3),
                },
                {
                    "指标": "Recall",
                    "数值": round(comparison["recall"], 3),
                },
                {
                    "指标": "F1",
                    "数值": round(comparison["f1"], 3),
                },
            ],
            hide_index=True,
            use_container_width=True,
        )
    with diff_right:
        st.dataframe(
            [
                {
                    "差异类型": "规则多切出的边界",
                    "位置索引": ", ".join(map(str, comparison["pred_only"])) or "无",
                },
                {
                    "差异类型": "规则漏掉的真实边界",
                    "位置索引": ", ".join(map(str, comparison["gold_only"])) or "无",
                },
            ],
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("为什么之前无法稳定抓到真实样本"):
        st.markdown(
            """
            - `raw.githubusercontent.com/.../data/rst/` 只能下载具体文件，不能像目录一样列出内容。
            - 之前的实现偏向“先遍历目录再猜格式”，一旦 GitHub API 或 raw 请求失败，就只能退回本地 fallback。
            - 现在改成了“先用已确认存在的公开样本对 `.out` + `.out.edus` 直接抓取”，并把失败原因原样展示出来。
            """
        )
        st.dataframe(
            [{"诊断信息": item} for item in neural_sample.get("diagnostics", [])],
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("### 为什么神经模型更稳？")
    st.markdown(
        """
        <div class="note-box">
            这里的规则基线主要依赖标点、从属连词和局部依存关系，因此容易把“逗号 + 从句”都当作边界，
            或者漏掉没有明显标记但语义上应切开的地方。课件 P36 提到的 restricted self-attention
            可以理解为：模型不会把注意力平均撒到整句的所有词上，而是更聚焦于边界附近及其局部上下文，
            从而在长句中更容易同时看到“当前连接词、主句谓词、相邻从句结构”这些关键信号。
            这就是为什么 BiLSTM-CRF + 受限自注意力通常比简单规则更能稳定恢复真实 EDU 边界。
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("数据解析细节"):
        st.write(
            {
                "download_url": neural_sample.get("download_url", ""),
                "boundary_indices": neural_sample["boundary_indices"],
                "edus": neural_sample["edus"],
            }
        )

    with st.expander("把当前输入文本也跑一遍规则切分"):
        if text:
            user_baseline = segment_discourse_rule_based(text, nlp)
            render_edu_cards(
                user_baseline["edus"],
                user_baseline["boundary_indices"],
                "pred",
                "boundary",
                user_baseline.get("boundary_reasons"),
            )
            st.dataframe(
                [
                    {
                        "edu_id": i + 1,
                        "text": edu,
                        "reason": user_baseline.get("boundary_reasons", [""] * len(user_baseline["edus"]))[i],
                    }
                    for i, edu in enumerate(user_baseline["edus"])
                ],
                hide_index=True,
                use_container_width=True,
            )


def render_module_two(text: str, nlp: Language) -> None:
    st.subheader("模块 2：浅层篇章分析与显式关系提取")
    default_sentence = (
        "Third-quarter sales in Europe were exceptionally strong, boosted by promotional "
        "programs and new products - although weaker foreign currencies reduced the company's earnings."
    )
    sentence = st.text_area(
        "输入一个包含显式连接词的英文句子",
        value=default_sentence,
        height=120,
        key="module2_sentence",
    )

    info = find_explicit_connective(sentence) if sentence else None

    if info:
        arg1, arg2 = split_arguments(sentence, info)
        st.markdown("### 连接词识别")
        st.markdown(
            f'<div class="compare-panel" style="line-height:1.9;">{highlight_connective(sentence, info)}</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric_card("连接词", info["display"], "句中最先识别到的显式 discourse connective")
        with col2:
            render_metric_card("PDTB 顶级类", info["label"], "根据规则表和局部语境得到的粗粒度类别")
        with col3:
            render_metric_card("Arg 切分", "Arg1 / Arg2", "以连接词为界做简化版论据提取")

        arg_left, arg_right = st.columns(2)
        with arg_left:
            st.markdown(
                f"""
                <div class="arg-box arg1-box">
                    <strong>Arg1</strong>
                    <div style="margin-top:0.6rem; line-height:1.8;">{html.escape(arg1)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with arg_right:
            st.markdown(
                f"""
                <div class="arg-box arg2-box">
                    <strong>Arg2</strong>
                    <div style="margin-top:0.6rem; line-height:1.8;">{html.escape(arg2)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        relation_row = {
            "connective": info["display"],
            "sense": info["label"],
            "arg1": arg1,
            "arg2": arg2,
            "disambiguation_note": info["note"] or "非歧义连接词，按规则表直接映射。",
        }
        st.dataframe([relation_row], hide_index=True, use_container_width=True)
    else:
        st.warning("当前句子里没有命中预设的显式连接词，请尝试输入 although / because / since / but / after 等。")

    st.markdown("### `since` 消歧观察任务")
    example_a = "Since 2020, the company has expanded into three new markets."
    example_b = "The company delayed the launch since demand had already fallen."
    since_sentence = st.selectbox(
        "选择课件式 `since` 例句进行观察",
        [example_a, example_b],
        key="since_example",
    )
    since_info = find_explicit_connective(since_sentence)
    if since_info:
        st.markdown(
            f'<div class="compare-panel" style="line-height:1.9;">{highlight_connective(since_sentence, since_info)}</div>',
            unsafe_allow_html=True,
        )
        st.info(
            f"`since` 当前被判为 `{since_info['label']}`。"
            f" 解释：{since_info['note'] or '这里使用的是固定映射。'}"
        )

    with st.expander("显式连接词规则表"):
        mapping_rows = [
            {"connective": conn, "coarse_label": label}
            for conn, label in EXPLICIT_CONNECTIVES.items()
        ]
        st.dataframe(mapping_rows, use_container_width=True, hide_index=True)

    with st.expander("工程观察"):
        st.markdown(
            """
            - `since` 的难点在于词形相同，但既可能表示时间起点，也可能表示原因。
            - 这里只用了很浅的上下文规则，所以在复杂句、倒装句、长距离依赖里仍然容易判错。
            - 真正的显式连接词消歧通常需要更丰富的句法、语义和上下文特征，而不仅仅是看连接词本身。
            """
        )


def render_module_three(text: str, nlp: Language) -> None:
    st.subheader("模块 3：指代消解（Coreference Resolution）可视化")
    default_text = (
        "Barack Obama was born in Hawaii. He was elected president in 2008. "
        "Obama said his administration would focus on healthcare, and it became a major priority. "
        "His supporters believed they could help him pass the reform."
    )
    coref_text = st.text_area(
        "输入包含多个人称代词的英文段落",
        value=default_text,
        height=180,
        key="module3_text",
    )
    clusters, message = run_coref(coref_text, nlp) if coref_text else ([], "")
    st.info(message or "请输入文本后查看指代结果。")

    if coref_text:
        st.markdown("### 原文高亮")
        if clusters:
            st.markdown(
                f'<div class="compare-panel" style="line-height:2.0;">{render_coref_text(coref_text, clusters)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="compare-panel" style="line-height:2.0;">{html.escape(coref_text)}</div>',
                unsafe_allow_html=True,
            )

    if clusters:
        st.markdown("### 指代簇列表")
        for cluster in clusters:
            mentions = " -> ".join(cluster["mentions"])
            st.markdown(
                f"""
                <div class="edu-box gold">
                    <strong>{html.escape(cluster["entity"])}</strong>
                    <span class="tag">{html.escape(cluster["method"])}</span>
                    <div style="margin-top: 0.45rem;">{html.escape(mentions)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.dataframe(
            [
                {
                    "cluster": cluster["entity"],
                    "mentions": cluster["mentions"],
                    "method": cluster["method"],
                }
                for cluster in clusters
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("当前文本中没有检测到稳定的多提及实体链。")

    st.markdown("### 观察任务")
    st.markdown(
        """
        - 尝试输入同时包含 `he / she / it / they` 的跨句段落，观察模型是否能把跨句回指聚到同一实体。
        - 从 mention ranking 的角度看，底层模型会为候选先行词和当前代词计算关联分数，再选择最可能的链接对象。
        - 这些分数通常会综合表述距离、句法位置、语义相容性和上下文表示，因此复杂代词仍然可能出错。
        """
    )


def main() -> None:
    inject_styles()
    text = ""
    nlp = load_spacy_model()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">DISCOURSE LAB</div>
            <h1>篇章分析综合平台</h1>
            <p>
                这个系统把话语分割、浅层篇章关系提取和指代消解放到同一个交互式页面中，
                方便从 “EDU 切分 - 连接词关系 - 实体链恢复” 三个层面直观看到篇章衔接与连贯是如何形成的。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["模块 1：话语分割", "模块 2：浅层篇章关系", "模块 3：指代消解"])
    with tab1:
        render_module_one(text, nlp)
    with tab2:
        render_module_two(text, nlp)
    with tab3:
        render_module_three(text, nlp)


if __name__ == "__main__":
    main()
