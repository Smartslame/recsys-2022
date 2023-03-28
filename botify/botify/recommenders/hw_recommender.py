import random

import numpy as np

from .random import Random
from .recommender import Recommender


class HwRecommender(Recommender):
    """
    Recommend tracks closest to the previous one.
    Fall back to the random recommender if no
    recommendations found for the track.
    """

    def __init__(self, tracks_redis, context_embeddings, track_embeddings, user_embeddings, k):
        self.tracks_redis = tracks_redis
        self.fallback = Random(tracks_redis)
        self.context_embeddings = context_embeddings
        self.track_embeddings = track_embeddings
        self.user_embeddings = user_embeddings
        self.k = k

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        if prev_track is None or prev_track >= self.context_embeddings.shape[0]:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        context_embedding = self.context_embeddings[prev_track]

        if user >= self.user_embeddings.shape[0]:
            return self.get_recomendation(np.dot(context_embedding, self.track_embeddings))

        user_embedding = self.user_embeddings[user]

        return self.get_recomendation(np.dot(context_embedding + user_embedding, self.track_embeddings.T))

    def get_recomendation(self, scores):
        neighbours = np.argpartition(-scores, self.k)[:self.k]
        shuffled = list(neighbours)
        random.shuffle(shuffled)
        return int(shuffled[0])
