import abc
import dataclasses
import random

import numpy as np


@dataclasses.dataclass
class ItemPool:
    mask: np.ndarray

    def masked_argmax(self, scores):
        scores = scores.copy()
        scores[~self.mask] = -np.inf
        return np.argmax(scores)

    def masked_argallmax(self, scores):
        scores = scores.copy()
        scores[~self.mask] = -np.inf

        best_score = np.max(scores)
        best_items = np.flatnonzero(scores == best_score)

        return random.choice(best_items)

    def items(self):
        return self.mask.nonzero()[0]

    @property
    def item_count(self):
        return self.mask.shape[0]


class Agent(abc.ABC):
    def __init__(self):
        self.item_features = None

    def set_features(self, features):
        self.item_features = features

    def reset(self, user_count, item_count):
        self.user_count = user_count
        self.item_count = item_count

    @abc.abstractmethod
    def act(self, user, pool: ItemPool):
        raise NotImplementedError

    def update(self, user, item, reward):
        pass

    @property
    def label(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.label


class Estimator:
    def reset(self, user_count, item_count):
        super().reset(user_count, item_count)

    def update(self, user, item, reward):
        pass

    @abc.abstractmethod
    def get_clicks(self, user):
        raise NotImplementedError

    @abc.abstractmethod
    def get_imps(self, user):
        raise NotImplementedError

    def get_ctrs(self, user):
        clicks = self.get_clicks(user)
        imps = self.get_imps(user)
        return np.divide(clicks, imps, where=imps > 0, out=np.zeros(imps.shape))


class EstimatorAgent(Agent):
    def __init__(self, estimator: Estimator):
        self.estimator = estimator

    def reset(self, user_count, item_count):
        super().reset(user_count, item_count)
        self.estimator.reset(user_count, item_count)
        self.t = 1

    def update(self, user, item, reward):
        self.t += 1
        self.estimator.update(user, item, reward)
