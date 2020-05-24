from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import pandas as pd
import numpy as np

movie_data = pd.read_csv('data.csv')


def Knn_function(movie_data):
    tfv = TfidfVectorizer(min_df=3, max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3),
                          stop_words='english')

    movie_data['overview'] = movie_data['overview'].fillna('')
    tfv_matrix = tfv.fit_transform(movie_data['overview'])
    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
    indices = pd.Series(movie_data.index, index=movie_data['original_title']).drop_duplicates()

    return sig, indices


sig, indices = Knn_function(movie_data)


def get_record(title, sig=sig):
    title = title.lower()
    if title not in indices:
        return "Not in Database"
    else:
        # get index of original title
        idx = indices[title]

        # Get the pairwsie similarity scores
        sig_scores = list(enumerate(sig[idx]))

        # Sort the movies
        sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

        # Sort the movies
        sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

        # Scores of the 10 most similar movies
        sig_scores = sig_scores[1:11]

        # Movie indices
        movie_indices = [i[0] for i in sig_scores]

        # Top 10 most similar movies
        return list(movie_data['original_title'].iloc[movie_indices])

