from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn import preprocessing

from utils.utils import clean_text


@dataclass
class MoviePreprocessor:

    @staticmethod
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Clean text fields
        df["clean_keywords"] = df["plot_keywords"].fillna("").apply(clean_text)
        df["clean_genres"] = df["genres"].fillna("").apply(lambda s: "|".join(sorted(set(str(s).split("|")))))
        df["clean_title"] = df["movie_title"].fillna("").str.strip()

        # Numeric safety
        for col in ["imdb_score", "num_voted_users", "budget", "title_year"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Normalize IMDb ratings to 0-1 scale
        scaler = preprocessing.MinMaxScaler()
        col = df["imdb_score"].astype(float)
        df["imdb_norm"] = scaler.fit_transform(col.values.reshape(-1, 1)).ravel()
        df["imdb_norm"] = df["imdb_norm"].astype(float)

        # One-hot encode genres (pipe-separated)
        mlb = preprocessing.MultiLabelBinarizer()
        genres_split = df["clean_genres"].fillna("").apply(lambda s: [t.strip() for t in str(s).split("|") if t.strip()]).tolist()
        genres_ohe = mlb.fit_transform(genres_split)
        genre_cols = [f"genre_{g.replace(' ', '_').lower()}" for g in mlb.classes_]
        df_genres = pd.DataFrame(genres_ohe, columns=genre_cols, index=df.index)
        for col in df_genres.columns:
            df[col] = df_genres[col].astype("uint8")
        del df_genres

        # Binary indicator encoding for selected
        def add_label_encoding(colname: str):
            if colname not in df.columns:
                return
            col = df[colname].fillna("").astype(str)
            if col.eq("").all():
                return
            le = preprocessing.LabelEncoder()
            df[f"le_{colname.lower()}"] = le.fit_transform(col).astype("int32")

        for cat in ["color", "language", "country", "content_rating"]:
            add_label_encoding(cat)

        return df
