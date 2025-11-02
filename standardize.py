#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, time, argparse, sys
import numpy as np
import pandas as pd
from collections import defaultdict

import faiss
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process
import torch

# -------------------
# Defaults (override via CLI)
# -------------------
DEFAULT_MODEL = 'pritamdeka/S-PubMedBert-MS-MARCO'
INDEX_FILE = "nucc_index.faiss"
MAP_FILE   = "nucc_map.json"   # stores rows, index_texts, and meta (model/dim)

TOPK_SEM     = 50    # semantic candidate pool per query
TOPK_FUZZY   = 100   # fuzzy candidate pool size (global)
CONF_SEM     = 0.40  # keep if semantic cosine >= this
CONF_FUZZY   = 60    # keep if fuzzy >= this (0..100)
FUSE_KEEP    = 0.50  # final fused threshold (0..1); set None to keep OR of above
BATCH_ENC    = 256   # encoding batch size
USE_GPU      = True  # auto-fallback to CPU if no CUDA

# Regex: NUCC codes are 10-char alphanumeric (e.g., 207ZP0104X, 261QX0203X)
NUCC_CODE_RE = re.compile(r"\b[A-Z0-9]{10}\b")

# Cleanup / splitting
SPLIT_RE = re.compile(r"\s*(?:[/,;+]|(?:\s*&\s*)|(?:\s+and\s+)|-|;)\s*")
CLEAN_RE = re.compile(r"[^a-z0-9\s\-]")

SAFE_JUNK = {
    "dept","department","of","clinic","center","centre","services","provider",
    "doc","doctor","physician","group","unit","care","office","practice","hospital","hosp"
}

def normalize(s: str) -> str:
    s = s.lower()
    s = s.replace("’","'")
    s = CLEAN_RE.sub(" ", s)
    toks = [t for t in s.split() if t not in SAFE_JUNK]
    return " ".join(toks).strip()

def split_parts(s: str):
    s = normalize(s)
    if not s:
        return []
    parts = [p for p in SPLIT_RE.split(s) if p]
    return parts if parts else [s]

def load_synonyms(path: str):
    """Expect columns: alias, canonical. canonical may contain ' | ' for multi-map."""
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    syn = {}
    for _, r in df.iterrows():
        a = normalize(str(r.get("alias","")))
        c = str(r.get("canonical","")).strip()
        if a and c:
            syn[a] = c
    return syn

# -------------------
# NUCC index building / loading (no NUCC mutation)
# -------------------
def _find_col(df, name):
    m = {c.lower(): c for c in df.columns}
    return m.get(name.lower(), None)

def _cuda_ok():
    return USE_GPU and torch.cuda.is_available()

def build_nucc_index(nucc_csv: str, model_name: str):
    """Create a semantic index over NUCC rows using a rich text view per code."""
    df = pd.read_csv(nucc_csv, dtype=str).fillna("")

    # Optional filter to active rows if Status exists
    c_status = _find_col(df, "Status")
    if c_status:
        mask = df[c_status].str.lower().str.contains("active", na=False)
        df = df.loc[mask].copy()

    c_code  = _find_col(df, "Code")
    c_disp  = _find_col(df, "Display_Name")
    c_class = _find_col(df, "Classification")
    c_spec  = _find_col(df, "Specialization")

    if not c_code:
        raise ValueError("NUCC CSV must include a 'Code' column.")

    rows = []
    index_texts = []
    for _, r in df.iterrows():
        code = str(r[c_code]).strip()
        disp = str(r[c_disp]).strip() if c_disp else ""
        clas = str(r[c_class]).strip() if c_class else ""
        spec = str(r[c_spec]).strip() if c_spec else ""

        # Rich index view: prefer Display_Name, then append missing bits
        pieces = []
        if disp: pieces.append(disp)
        if clas and clas not in disp: pieces.append(clas)
        if spec and spec not in disp: pieces.append(spec)
        view = " | ".join(pieces) if pieces else code

        index_texts.append(view)
        rows.append({
            "code": code,
            "display_name": disp,
            "classification": clas,
            "specialization": spec
        })

    device = "cuda" if _cuda_ok() else "cpu"
    print(f"Encoding NUCC ({len(index_texts)}) with {model_name} on {device} ...")
    encoder = SentenceTransformer(model_name, device=device)
    embs = encoder.encode(index_texts, show_progress_bar=True, batch_size=BATCH_ENC,
                          convert_to_numpy=True).astype("float32")

    # Cosine similarity via normalized dot product
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    # Persist with meta
    faiss.write_index(index, INDEX_FILE)
    meta = {
        "rows": rows,
        "index_texts": index_texts,
        "model_name": model_name,
        "dim": int(embs.shape[1]),
        "built_at": time.time(),
    }
    with open(MAP_FILE, "w") as f:
        json.dump(meta, f)

    return index, rows, index_texts, encoder

def load_nucc_index(model_name: str):
    if not (os.path.exists(INDEX_FILE) and os.path.exists(MAP_FILE)):
        return None, None, None, None
    index = faiss.read_index(INDEX_FILE)
    with open(MAP_FILE, "r") as f:
        data = json.load(f)

    # Rebuild if model or dim mismatch
    if (data.get("model_name") != model_name) or (index.d != int(data.get("dim", -1))):
        print("Index/model mismatch detected — will rebuild.")
        return None, None, None, None

    device = "cuda" if _cuda_ok() else "cpu"
    encoder = SentenceTransformer(model_name, device=device)
    return index, data["rows"], data["index_texts"], encoder

# -------------------
# Candidate gathering (semantic + fuzzy + fuse)
# -------------------
def gather_candidates(term, index, index_texts, rows, encoder):
    """
    Return list of dicts:
    {code, src, sem, fuz, fuse}
    """
    out = {}

    # 1) Semantic topK
    q = encoder.encode([term], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)
    D, I = index.search(q, TOPK_SEM)
    for j in range(len(I[0])):
        idx = int(I[0][j])
        if idx == -1:
            continue
        sem = float(D[0][j])
        if sem < CONF_SEM:
            continue
        code = rows[idx]["code"]
        cur = out.get(code, {"code": code, "src": index_texts[idx], "sem":0.0, "fuz":0.0})
        cur["sem"] = max(cur["sem"], sem)
        out[code] = cur

    # 2) Fuzzy topK (global)
    fuzzy_hits = process.extract(term, index_texts, scorer=fuzz.WRatio, limit=TOPK_FUZZY)
    for s, score, idx in fuzzy_hits:
        if score < CONF_FUZZY:
            continue
        code = rows[idx]["code"]
        cur = out.get(code, {"code": code, "src": index_texts[idx], "sem":0.0, "fuz":0.0})
        cur["fuz"] = max(cur["fuz"], float(score)/100.0)
        out[code] = cur

    # 3) Fuse & keep
    kept = []
    for d in out.values():
        fuse = 0.5*d["sem"] + 0.5*d["fuz"]
        d["fuse"] = fuse
        if FUSE_KEEP is None:
            kept.append(d)
        else:
            if fuse >= FUSE_KEEP or d["sem"] >= CONF_SEM or d["fuz"] >= (CONF_FUZZY/100.0):
                kept.append(d)

    return kept

# -------------------
# CLI argparse (strict, required flags)
# -------------------
def parse_args():
    parser = argparse.ArgumentParser(description="NUCC Specialty Standardizer (hybrid, multi-code)")
    parser.add_argument("--nucc", required=True,
                        help="Path to NUCC master CSV, e.g. nucc_taxonomy_master.csv")
    parser.add_argument("--input", required=True,
                        help="Path to input CSV with column 'raw_specialty'")
    parser.add_argument("--out", required=True,
                        help="Output CSV path")
    parser.add_argument("--synonyms", default=None,
                        help="Optional synonyms CSV (alias,canonical)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Sentence-Transformer model (default: {DEFAULT_MODEL})")
    return parser.parse_args()

# -------------------
# Main
# -------------------
def main():
    args = parse_args()

    t0 = time.time()
    # Load index or build
    index, rows, index_texts, encoder = load_nucc_index(args.model)
    if index is None:
        index, rows, index_texts, encoder = build_nucc_index(args.nucc, args.model)

    # Quick lookups
    code_to_row = {r["code"]: r for r in rows}
    codes_set   = set(code_to_row.keys())

    # Synonyms
    synonyms = load_synonyms(args.synonyms)
    if synonyms:
        print(f"Loaded {len(synonyms)} synonyms.")

    # Read input
    df_in = pd.read_csv(args.input, dtype=str).fillna("")
    if "raw_specialty" not in df_in.columns:
        raise ValueError("INPUT CSV must have a 'raw_specialty' column.")
    out_rows = []

    for _, r in df_in.iterrows():
        raw = str(r["raw_specialty"])
        explanations = []
        agg_codes = set()
        confidences = []

        # Stage 0: direct NUCC code capture (works for 'RADIATION - 261QX0203X')
        direct_codes = {m.group(0).upper() for m in NUCC_CODE_RE.finditer(raw.upper())}
        direct_codes = {c for c in direct_codes if c in codes_set}
        if direct_codes:
            agg_codes |= direct_codes
            for c in direct_codes:
                explanations.append(f"Direct NUCC code '{c}' detected in input.")
                confidences.append(1.0)  # max confidence

        # Stage 1: segments (/, &, +, -, and, commas, semicolons)
        parts = split_parts(raw)
        if not parts and not agg_codes:
            out_rows.append({
                "raw_specialty": raw, "nucc_codes": "JUNK", "confidence": 0.0,
                "explain": "Input was empty or only boilerplate"
            })
            continue

        # Stage 2: synonym expansion (alias -> canonical, allow multi via '|')
        expanded = []
        for p in parts:
            if not p:
                continue
            if p in synonyms:
                canon = synonyms[p]
                # allow multi-map e.g. "pulm/crit" -> "pulmonary disease | critical care medicine"
                for seg in [x.strip() for x in canon.split("|") if x.strip()]:
                    expanded.append(seg)
                    explanations.append(f"Synonym '{p}' → '{seg}'.")
            else:
                expanded.append(p)

        # Stage 3: gather candidates for each expanded part
        for term in expanded:
            if not term:
                continue
            cand = gather_candidates(term, index, index_texts, rows, encoder)
            if not cand:
                explanations.append(f"No confident candidate for '{term}'.")
                continue
            for d in cand:
                agg_codes.add(d["code"])
                confidences.append(d.get("fuse", max(d.get("sem",0.0), d.get("fuz",0.0))))
                explanations.append(f"'{term}' → {d['code']} via sem={d.get('sem',0):.2f}, fuzzy={d.get('fuz',0):.2f} :: {d['src'][:90]}")

        if not agg_codes:
            out_rows.append({
                "raw_specialty": raw, "nucc_codes": "JUNK", "confidence": 0.0,
                "explain": "; ".join(explanations)[:400]
            })
            continue

        # Confidence: conservative — min of top-3 signals
        conf = 0.0
        if confidences:
            confidences = sorted(confidences, reverse=True)[:3]
            conf = min(confidences)

        out_rows.append({
            "raw_specialty": raw,
            "nucc_codes": " | ".join(sorted(agg_codes)),
            "confidence": round(float(conf), 4),
            "explain": "; ".join(explanations)[:400]
        })

    pd.DataFrame(out_rows).to_csv(args.out, index=False)
    print(f"Done. Wrote {len(out_rows)} rows to {args.out} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
