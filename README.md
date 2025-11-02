# Provider Specialty Standardization — NUCC Taxonomy Mapping

## Overview
This repository contains my submission for the **Provider Specialty Standardization Challenge** by HiLabs.  
The project aims to standardize unstructured healthcare provider specialty text fields (e.g., “Cardio”, “ENT Surgeon”, “Addiction Med.”) into official **NUCC Taxonomy Codes** — a federally maintained classification standard by CMS and AMA.

The provided solution builds an efficient and deterministic end-to-end mapping pipeline that runs locally and processes **20,000+ rows in under 15 minutes**, using a hybrid of **semantic embeddings**, **fuzzy string matching**, and **rule-based synonym expansion**.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/aarchit22/Alpha-Nexus.git
cd nucc-standardizer
```

### 2. Create Environment and Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate   # (on Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

If you do not have a `requirements.txt`, you can manually install dependencies:
```bash
pip install numpy pandas torch faiss-cpu sentence_transformers rapidfuzz
```

### 3. Prepare Input Data
- Place your **NUCC taxonomy file** (e.g., `nucc_taxonomy_master.csv`) in the working directory.  
- Create an **input CSV** (e.g., `input_specialties.csv`) with a single column `raw_specialty` containing raw specialty text.  
- Optionally, provide a **synonyms.csv** mapping file (with columns `alias,canonical`) for short forms and abbreviations.

### 4. Run the Script
```bash
python standardize.py --nucc nucc_taxonomy_master.csv --input input_specialties.csv --out output.csv
```

### 5. Output
The script generates an `output.csv` file with the following columns:
| Column | Description |
|---------|--------------|
| **raw_specialty** | Original input string |
| **nucc_codes** | Pipe-separated list of taxonomy codes or `JUNK` |
| **confidence** | Confidence score (0–1) |
| **explain** | Summary of mapping rationale |

---

## Preprocessing Logic

### 1. Text Normalization
- Converts text to lowercase  
- Removes punctuation and symbols  
- Strips boilerplate words like “dept”, “clinic”, “provider”, etc.  
- Collapses extra whitespace  
- Uses regex to directly detect embedded NUCC codes (10-character alphanumeric pattern)

### 2. Synonym Expansion
- Loads optional `synonyms.csv` with mappings like:  
  ```csv
  alias,canonical
  ENT,Otolaryngology
  OBGYN,Obstetrics & Gynecology
  Cardio,Cardiology
  ```
- Each alias is replaced with its canonical equivalent before matching.

### 3. Specialty Segmentation
- Splits composite entries (e.g., `Cardio/Diab`, `Pain + Spine`, `Surgery and Ortho`) using delimiters: `/`, `+`, `&`, `and`, `;`, `-`.
- Each segment is processed individually and mapped separately.

---

## Mapping Approach

### 1. Semantic Search (Vector Similarity)
- Uses a biomedical SentenceTransformer model (`pritamdeka/S-PubMedBert-MS-MARCO`) to encode all NUCC taxonomy text fields (`Display_Name`, `Classification`, `Specialization`).
- The embeddings are indexed with **FAISS** for efficient cosine similarity search.
- For each input, top semantic matches are retrieved above a confidence threshold (default 0.4).

### 2. Fuzzy Matching
- Uses `rapidfuzz` WRatio to compute edit-distance–based similarity against all taxonomy entries.
- Keeps matches above a fuzzy threshold (default 60/100).

### 3. Score Fusion
- Combines semantic and fuzzy scores:  
  `fuse_score = 0.5 * semantic + 0.5 * fuzzy`
- Keeps all codes above a threshold (default 0.5).

### 4. Final Decision
- If one or more taxonomy codes meet criteria → output all joined by ` | `  
- If none meet threshold → mark as `JUNK`  
- Confidence is conservatively computed as the minimum of top three match scores.

---

## Performance
| System | Runtime (20K rows) | GPU | Memory |
|---------|-------------------|------|---------|
| RTX 3060 (6GB) | ~13 min | CUDA enabled | 4.5 GB |
| CPU (8-core) | ~25 min | No | 6 GB |

- Deterministic output — same input → same result.  
- Handles abbreviations, short forms, and partial matches effectively.

---

## Example

### Input (`input_specialties.csv`)
```
raw_specialty
Cardio
OBGYN
Pain & Spine Doc
Something random
```

### Output (`output.csv`)
```
raw_specialty,nucc_codes,confidence,explain
Cardio,207RC0000X,0.86,"Matched via semantic: cardiology"
OBGYN,207V00000X,0.91,"Synonym expansion: OBGYN → Obstetrics & Gynecology"
Pain & Spine Doc,208VP0014X | 207LP2900X,0.73,"Multi-specialty split and mapping"
Something random,JUNK,0.00,"No confident mapping found"
```

---

## Optional Spell Correction (Extension)
An optional transformer-based correction layer can be inserted before normalization to fix spelling errors.  
Potential pretrained models include:
- [NeuSpell](https://github.com/neuspell/neuspell)
- [SAGE: Transformer Spell Corrector](https://github.com/ai-forever/sage)

These can correct errors like “Anesthesiolgy” → “Anesthesiology” before embedding.

---

## Folder Structure
```
.
├── standardize.py
├── nucc_taxonomy_master.csv
├── input_specialties.csv
├── output.csv
├── synonyms.csv
├── README.md
└── requirements.txt
```

---

