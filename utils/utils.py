from __future__ import annotations

import random
import re

import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords

english_stopwords = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """
    Basic text cleaning: lowercasing, removing HTML tags, non-alphanumeric characters, extra spaces, stopwords.
    :param text:
    :return:
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Tokenize and remove stopwords
    toks = [t for t in text.split() if t and t not in english_stopwords]
    text = " ".join(toks)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def set_seed(seed: int = 19) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def to_user_item_matrix(ratings_df: pd.DataFrame, user_col: str, item_col: str, rating_col: str):
    users = ratings_df[user_col].astype("category")
    items = ratings_df[item_col].astype("category")
    mat = np.zeros((users.cat.categories.size, items.cat.categories.size), dtype=float)
    mat[:, :] = np.nan
    mat[users.cat.codes, items.cat.codes] = ratings_df[rating_col].values
    return mat, users.cat.categories.tolist(), items.cat.categories.tolist()


