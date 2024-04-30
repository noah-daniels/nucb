import numpy as np


class IncrementalCosine2:
    def __init__(self, user_count, item_count):
        self.values = np.zeros((user_count, item_count))
        self.scores = np.zeros((user_count, user_count))
        self.mags = np.zeros(user_count)
        self.sims = np.eye(user_count)

    @property
    def user_count(self):
        return self.values.shape[0]

    def update(self, user, item, value):
        # update internals
        self.mags[user] += 2 * self.values[user, item] * value + value * value
        self.scores[user, :] += value * self.values[:, item]
        self.scores[:, user] += value * self.values[:, item]
        self.values[user, item] += value

        # calculate new similarities
        base = np.sqrt(np.multiply(self.mags[user], self.mags))
        sims = np.divide(
            self.scores[user], base, where=base > 0, out=np.zeros(self.user_count)
        )
        self.sims[user, :] = sims
        self.sims[:, user] = sims
        self.sims[user, user] = 1

    def set(self, user, item, value):
        change = value - self.values[user, item]
        if abs(change) > 1e-8:
            self.update(user, item, change)


class IncrementalCosine3:
    def __init__(self, user_count, feature_count):
        self.values = np.zeros((user_count, feature_count))
        self.scores = np.zeros((user_count, user_count))
        self.mags = np.zeros(user_count)
        self.sims = np.eye(user_count)

    @property
    def user_count(self):
        return self.values.shape[0]

    def set_user(self, user, values):
        self.values[user] = values
        self.mags[user] = np.power(self.values[user], 2).sum()
        new_scores = self.values @ values
        self.scores[user, :] = new_scores
        self.scores[:, user] = new_scores

        base = np.sqrt(np.multiply(self.mags[user], self.mags))
        sims = np.divide(
            self.scores[user], base, where=base > 0, out=np.zeros(base.shape)
        )
        self.sims[user, :] = sims
        self.sims[:, user] = sims
        self.sims[user, user] = 1
