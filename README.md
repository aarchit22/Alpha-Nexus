# NUCC Specialty Standardizer (HiLabs Challenge)

This repository contains a fast, deterministic Python pipeline to standardize raw provider specialties to NUCC Taxonomy codes. It is designed for the HiLabs “Provider Specialty Standardization” challenge and follows the rules:

- No external APIs (everything is local).
- ≤ 15 minutes runtime on ~20k rows (typical Colab T4/A100 or modest workstation).
- Deterministic outputs given the same inputs and thresholds.
- Returns all plausible NUCC codes for ambiguous inputs (pipe-separated), or `JUNK` if confidence is below threshold.

---

## What this tool does

Given:

- `nucc_taxonomy_master.csv` (official NUCC master),
- `input_specialties.csv` with a column `raw_specialty`,
- `synonyms.csv` (optional custom mapping of slang/abbrev → canonical phrases),

the script:

1. Builds/loads a **semantic FAISS** index of NUCC specialties using a biomedical sentence embedding model (`pritamdeka/S-PubMedBert-MS-MARCO` by default).
2. **Normalizes and splits** each `raw_specialty` into parts (handles `/`, `&`, `and`, commas, dashes, etc.).
3. **Expands parts** through an optional synonym map (multi-mapping supported via `|`).
4. For each part, retrieves candidates using:
   - **Semantic search** (Sentence-Transformers embeddings + FAISS, cosine/IP).
   - **Fuzzy string matching** (RapidFuzz WRatio).
5. **Fuses** semantic and fuzzy scores, filters by thresholds, and aggregates all matching codes per input row.
6. Writes `output.csv` with:
   - `raw_specialty`
   - `nucc_codes` (pipe-separated codes, or `JUNK`)
   - `confidence` (0–1 float)
   - `explain` (compact rationale for mapping)

---

## Core logic

### Preprocessing and splitting

- Lowercase, remove punctuation, strip boilerplate tokens (e.g., `dept`, `clinic`, `center`, `provider`, etc.).
- Split composite strings: `Cardio / Endo & Diab` → `["cardio", "endo", "diab"]`.
- Optional synonym expansion (per part) using `synonyms.csv`:
  - CSV columns: `alias,canonical`
  - `canonical` may contain multiple canonical phrases separated by `|`.
  - Example:
    ```csv
    alias,canonical
    obgyn,obstetrics | gynecology
    ent,otolaryngology
    pulm/crit,pulmonary disease | critical care medicine
    ```

### Candidate gathering

For each (expanded) part:

- **Semantic Top-K** via FAISS  
  Encoder: `pritamdeka/S-PubMedBert-MS-MARCO`  
  Similarity: cosine (via L2-normalized embeddings + inner product)  
  Keep matches with semantic score ≥ `CONF_SEM`.

- **Fuzzy Top-K** via RapidFuzz  
  Scorer: `WRatio`  
  Keep matches with fuzzy score ≥ `CONF_FUZZY` (0–100).

### Score fusion and filtering

- Per NUCC candidate, fuse as:
fused = 0.5 * semantic + 0.5 * fuzzy

- Keep a candidate if any of:
- `fused >= FUSE_KEEP`, or
- `semantic >= CONF_SEM`, or
- `fuzzy >= CONF_FUZZY/100`.

- Aggregate all kept candidate codes for the row.

### Direct NUCC code shortcut

- If the input string contains a valid NUCC code (`^[A-Z0-9]{10}$`) and it exists in the NUCC master, include it immediately with max confidence.

### Confidence score in output

- A conservative row-level confidence is computed as the **minimum of the top-3** fused scores observed for that row (direct code hits contribute 1.0).
- If no code survives thresholds, output `JUNK` with `confidence=0.0`.

---

## Repository structure

.
├── standardize.py # main script
├── requirements.txt # pinned minimal dependencies
├── README.md # this file
├── nucc_taxonomy_master.csv # (place here)
├── input_specialties.csv # (place here)
└── synonyms.csv # optional (alias,canonical)


On first run the script will also create:
- `nucc_index.faiss` – FAISS vector index of the NUCC corpus
- `nucc_map.json` – metadata (rows, index texts, model name, embedding dim)

These cache files allow fast subsequent runs without rebuilding the index.

---

## Setup

### 1) Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

2) Install dependencies
pip install -r requirements.txt


Notes:

faiss-cpu is used by default and is sufficient.

PyTorch will use GPU automatically if available; otherwise it falls back to CPU.


How to run

Basic usage:

python standardize.py --nucc nucc_taxonomy_master.csv --input input.csv --out output.csv


Thresholds and calibration

Defaults (from standardize.py):
TOPK_SEM   = 50
TOPK_FUZZY = 100
CONF_SEM   = 0.40
CONF_FUZZY = 60
FUSE_KEEP  = 0.50
BATCH_ENC  = 256
DEFAULT_MODEL = 'pritamdeka/S-PubMedBert-MS-MARCO'
