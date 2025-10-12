import random

import pandas as pd

seed = 7

df_5k = pd.read_csv('../data/movie_metadata.csv')
df_50k = pd.read_csv('../data/IMDB Dataset.csv')

positive_reviews = df_50k[df_50k['sentiment'] == 'positive']['review'].tolist()
negative_reviews = df_50k[df_50k['sentiment'] == 'negative']['review'].tolist()
random.seed(seed)


def get_random_review(is_positive: bool) -> str:
    if is_positive:
        return random.choice(positive_reviews)
    else:
        return random.choice(negative_reviews)


df_5k['review'] = df_5k['imdb_score'].apply(lambda x: get_random_review(x >= 7.0))
df_5k['sentiment'] = df_5k['imdb_score'].apply(lambda x: 'positive' if x >= 7.0 else 'negative')

df_5k.to_csv('movie_metadata_with_reviews.csv', index=False)
