from __future__ import annotations

import os
import warnings
from typing import cast

import numpy as np
import pandas as pd
from transformers import logging

from configuration.config import SentimentConfig, HybridWeights, EvalConfig, SystemConfig, Paths
from dataset.preprocessing import MoviePreprocessor
from models.evaluation import RankingMetrics, cross_validate_sentiment_kfold, ab_test_simulation, qualitative_topk_for_users
from models.sentiment import build_sentiment_analyzer
from recommenders.collaborative import CollaborativeRecommender
from recommenders.content_base import ContentRecommender
from recommenders.hybrid import HybridRecommender
from utils.utils import set_seed, clean_text

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.set_verbosity_error()


def extract_balanced_reviews_and_ratings(movies: pd.DataFrame):
    df = movies.copy()

    # Clean review text, map labels
    texts_all = df["review"].fillna("").astype(str).map(clean_text)
    sent_raw = df["sentiment"].fillna("").astype(str).str.lower().str.strip()
    label_map = {"positive": 1, "negative": 0}
    labels_all = sent_raw.map(label_map)

    mask_valid = labels_all.isin([0, 1]) & (texts_all.str.len() > 0)
    texts_all = texts_all[mask_valid]
    labels_all = labels_all[mask_valid]

    # Balance classes if possible
    if len(labels_all) > 0 and labels_all.dropna().isin([0, 1]).all():
        tmp = pd.DataFrame({"text": texts_all.values, "label": labels_all.values})
        vc = tmp["label"].value_counts()
        if len(vc) == 2:
            n = int(vc.min())
            pos = tmp[tmp.label == 1].sample(n=n, random_state=42, replace=False)
            neg = tmp[tmp.label == 0].sample(n=n, random_state=42, replace=False)
            tmp = pd.concat([pos, neg], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
        texts_bal = tmp["text"].tolist()
        labels_bal = tmp["label"].astype(int).tolist()
    else:
        texts_bal = texts_all.tolist()
        labels_bal = labels_all.astype(int).tolist()

    # Simulate user ratings for collaborative filtering (0–10 scale)
    user_ids = ["user1", "user2", "user3", "user4", "user5"]
    titles = df["clean_title"].fillna(df.get("movie_title")).astype(str).str.strip().tolist()

    ratings_data = []
    for user_id in user_ids:
        if not titles:
            continue
        n_ratings = int(np.random.randint(5, 11))
        idx_choices = np.arange(len(titles))
        n_ratings = min(n_ratings, len(idx_choices))
        user_movie_indices = np.random.choice(idx_choices, size=n_ratings, replace=False)
        for idx in user_movie_indices:
            movie_title = titles[idx]
            imdb_score = df.iloc[idx].get('imdb_score', 7.0)
            if pd.isna(imdb_score):
                imdb_score = 7.0
            # 0–10 base with user noise
            base_rating = float(imdb_score)
            user_noise = np.random.normal(0, 1.0)
            rating = np.clip(base_rating + user_noise, 0, 10)
            rating = float(rating)
            ratings_data.append({"user_id": user_id, "movie_title": movie_title, "rating": rating})

    ratings = pd.DataFrame(ratings_data)
    return texts_bal, labels_bal, ratings


def train_val_test_split_texts(texts, labels, train=0.7, val=0.15, seed=19):
    """
    Simple random split of texts and labels into train/val/test sets, 70% train, 15% val, 15% test by default.
    :param texts:
    :param labels:
    :param train:
    :param val:
    :param seed:
    :return:
    """
    idx = np.arange(len(texts))
    np.random.shuffle(idx)
    n = len(idx)
    n_train = int(n * train)
    n_val = int(n * val)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    def pick(ix):
        return [texts[i] for i in ix], [int(labels[i]) for i in ix]

    return pick(train_idx), pick(val_idx), pick(test_idx)


def main():
    paths = Paths()
    sent_cfg = SentimentConfig()
    system_cfg = SystemConfig()
    set_seed(system_cfg.seed)

    movies = pd.read_csv(paths.DATASET_PATH)

    pre = MoviePreprocessor()
    movies_pp = pre.transform(movies)
    sent_model = build_sentiment_analyzer(cfg=sent_cfg, system_cfg=system_cfg)

    content = ContentRecommender(max_features=20000)
    cf = CollaborativeRecommender(algo="svd", n_factors=16, n_iters=8)

    # Use reviews and sentiments from dataset
    review_texts, review_labels, ratings_df = extract_balanced_reviews_and_ratings(movies_pp)

    # Stratified K-fold cross-validation (k=5)
    if len(review_texts) and len(review_labels):
        cv = cross_validate_sentiment_kfold(review_texts, review_labels, sent_cfg, system_cfg, k=5, seed=system_cfg.seed)
        mean = cast(dict, cv["mean"])  # typing hint for static analyzers
        print("\n[Sentiment K-Fold CV] k=5")
        print(f"Accuracy={mean['accuracy']:.3f}  Precision={mean['precision']:.3f}  Recall={mean['recall']:.3f}  F1={mean['f1']:.3f}")

    # split for sentiment
    (x_tr, y_tr), (x_val, y_val), (x_te, y_te) = train_val_test_split_texts(review_texts, review_labels)
    # Train on train, validate on val
    sent_model.fit(x_tr, y_tr, val_texts=x_val, val_labels=y_val)
    # Evaluate on test
    te_probs = getattr(sent_model, "predict_proba")(x_te)
    te_pred = (te_probs[:, 1] >= 0.5).astype(int)
    # Accuracy
    acc = float(np.mean(np.array(te_pred) == np.array(y_te, dtype=int))) if len(y_te) else 0.0
    print(f"Sentiment test accuracy: {acc:.3f}")

    weights = HybridWeights(w_content=0.4, w_cf=0.4, w_sentiment=0.2)
    hybrid = HybridRecommender(content, cf, sent_model, weights)
    hybrid.fit(movies_pp, ratings_df, x_tr, y_tr, x_val, y_val)

    # Save trained model
    model_dir = paths.MODELS_DIR / "hybrid"
    hybrid.save(model_dir)
    print(f"Saved hybrid model to: {model_dir}")

    seed_titles = [movies_pp["clean_title"].iloc[0]]
    recs = hybrid.recommend_for_user(user_id="user1", k=5, seed_titles=seed_titles)

    print("\nTop-5 Hybrid Recommendations for user 1 (probability in [0,1]):")
    for rank, (title, score) in enumerate(recs, start=1):
        print(f"{rank:>2}. {title}  (prob={score:.3f})")

    # Evaluate ranking metrics for user1 (use threshold >= 7)
    print("\nRanking metrics for user1:")
    u1 = ratings_df[ratings_df["user_id"] == "user1"]
    relevant = {t for t, r in u1[["movie_title", "rating"]].itertuples(index=False) if r >= 7}
    rec_titles = [t for t, _ in recs]
    for k in EvalConfig().ks:
        p = RankingMetrics.precision_at_k(rec_titles, relevant, k)
        r = RankingMetrics.recall_at_k(rec_titles, relevant, k)
        print(f"Precision@{k}={p:.3f}  Recall@{k}={r:.3f}")

    # Leave-one-out evaluation and metrics
    ks = EvalConfig().ks

    def build_content():
        return ContentRecommender(max_features=20000)

    def build_cf():
        return CollaborativeRecommender(algo="svd", n_factors=16, n_iters=5)

    def build_hybrid():
        return HybridRecommender(ContentRecommender(max_features=20000), CollaborativeRecommender(algo="svd", n_factors=16, n_iters=5), sent_model, weights)

    # Evaluation
    print("\n[Evaluation] Comparing Content, CF, and Hybrid...")
    results = ab_test_simulation(movies_pp, ratings_df, {"content": build_content, "collaborative": build_cf, "hybrid": build_hybrid}, ks=ks, rating_threshold=7, max_heldout_per_user=3)
    for name, metrics in results.items():
        lines = [f"{k}={v:.3f}" for k, v in metrics.items() if k.startswith("precision@") or k.startswith("recall@") or k.startswith("map@") or k == "rmse"]
        print(f"{name.capitalize()}:  " + "  ".join(lines))

    # Fit standalone content and CF once for qualitative inspection
    content.fit(movies_pp)
    cf.fit(ratings_df)
    sample_users = list(dict.fromkeys(ratings_df["user_id"].astype(str).tolist()))[:3]
    qual = qualitative_topk_for_users(ratings_df, {"content": content, "collaborative": cf, "hybrid": hybrid}, sample_users, k=10)

    print("\n[Qualitative] Top-10 recommendations for sample users:")
    for user, per_model in qual.items():
        print(f"\nUser: {user}")
        for model_name, rec_list in per_model.items():
            print(f"  {model_name}:")
            # Clarify score type by model
            if model_name == "hybrid":
                suffix = "prob"
            elif model_name == "content":
                suffix = "sim"
            elif model_name == "collaborative":
                suffix = "score"
            else:
                suffix = "score"
            for rank, (title, score) in enumerate(rec_list[:10], start=1):
                print(f"    {rank:>2}. {title}  ({suffix}={score:.3f})")


if __name__ == "__main__":
    main()
