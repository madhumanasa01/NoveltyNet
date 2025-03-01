# app/corpus/loader.py

import gzip
import json
import os
import shutil
from typing import List, Dict

from app.config import settings


REQUIRED_FIELDS = {"id", "abstract"}


def ensure_papers_file() -> bool:
    """
    The repo ships the corpus as papers.json.gz (GitHub's file-size limit).
    Decompress it to papers.json on first use so every other code path can
    keep reading and writing plain JSON.
    """
    if os.path.exists(settings.PAPERS_PATH):
        return True
    packed = settings.PAPERS_PATH + ".gz"
    if not os.path.exists(packed):
        return False
    with gzip.open(packed, "rb") as src, open(settings.PAPERS_PATH, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return True


def load_papers() -> List[Dict]:
    """
    Loads paper metadata from a local JSON file.
    Required fields per paper: id, abstract
    Optional fields: title, year, concepts, authors, url
    """

    if not ensure_papers_file():
        return []

    with open(settings.PAPERS_PATH, "r", encoding="utf-8") as f:
        papers = json.load(f)

    if not isinstance(papers, list):
        raise ValueError("papers.json must contain a list")

    cleaned = []

    for p in papers:
        if not REQUIRED_FIELDS.issubset(p.keys()):
            continue  # skip malformed entries silently

        abstract = p["abstract"].strip() if p.get("abstract") else ""
        if not abstract:
            continue  # skip empty abstracts

        cleaned.append({
            "id": p["id"],
            "title": p.get("title", ""),
            "abstract": abstract,
            "year": int(p["year"]) if p.get("year") else None,
            "concepts": p.get("concepts", []),
            "authors": p.get("authors", []),
            "url": p.get("url"),
        })

    return cleaned


def save_papers(papers: List[Dict]) -> None:
    os.makedirs(os.path.dirname(settings.PAPERS_PATH), exist_ok=True)
    with open(settings.PAPERS_PATH, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
