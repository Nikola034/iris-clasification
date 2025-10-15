from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from configuration.config import Paths
from recommenders.hybrid import HybridRecommender


def _canonicalize_seeds(available_titles: List[str], seeds: Iterable[str]) -> List[str]:
    """
    Given a list of available titles and user-provided seed titles, return a list of canonical titles
    """
    title_set_lower = {t.lower(): t for t in available_titles}
    matched: List[str] = []

    for raw in seeds:
        s = (raw or "").strip()
        if not s:
            continue
        s_low = s.lower()
        # exact
        if s_low in title_set_lower:
            canonical = title_set_lower[s_low]
            matched.append(canonical)
            continue
        # contains
        found = None
        for t in available_titles:
            if s_low in t.lower():
                found = t
                break
        if found:
            matched.append(found)
        else:
            print(f"[warn] Seed title not found in model titles: '{s}' — skipping")

    seen = set()
    out: List[str] = []
    for t in matched:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def load_model(model_dir: Path | None = None) -> HybridRecommender:
    paths = Paths()
    p = Path(model_dir) if model_dir is not None else (paths.MODELS_DIR / "hybrid")
    if not p.exists():
        raise FileNotFoundError(f"Model directory not found: {p}. Train and save the model by running main.py first.")
    model = HybridRecommender.load(p)
    return model


def recommend_movies(user_id: str, k: int = 10, seed_titles: Iterable[str] = (), model_dir: Path | None = None):
    model = load_model(model_dir)
    seeds = list(seed_titles)
    if seeds:
        seeds = _canonicalize_seeds(getattr(model, "_titles", []), seeds)
        if not seeds:
            print("[info] None of the provided seeds matched; proceeding without seeds.")
    recs = model.recommend_for_user(user_id=user_id, k=k, seed_titles=seeds)
    return recs


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Movie recommendation inference using saved Hybrid model")
    p.add_argument("--user", required=False, default="user1", help="User ID for recommendations")
    p.add_argument("--k", type=int, default=10, help="Number of recommendations to return")
    p.add_argument("--seed", dest="seeds", action="append", default=None, help="Seed title to bias content-based similarity")
    p.add_argument("--model-dir", type=str, default=None, help="Path to saved model directory (defaults to Paths.MODELS_DIR/hybrid)")
    p.add_argument("--json", dest="as_json", action="store_true", help="Print results as JSON array")
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    seeds = args.seeds or []
    model_dir = Path(args.model_dir) if args.model_dir else None

    recs = recommend_movies(user_id=args.user, k=args.k, seed_titles=seeds, model_dir=model_dir)

    if args.as_json:
        print(json.dumps([{"title": t, "score": s} for t, s in recs], ensure_ascii=False))
    else:
        print(f"Top-{args.k} recommendations for user '{args.user}':")
        for i, (t, s) in enumerate(recs, start=1):
            print(f"{i:>2}. {t}  (score={s:.3f})")


if __name__ == "__main__":
    main()
