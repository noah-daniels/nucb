import numpy as np

from banditrec.agent import Estimator


class CTREstimator(Estimator):
    def reset(self, user_count, item_count):
        self.clicks = np.zeros(item_count)
        self.imps = np.zeros(item_count)

    def update(self, user, item, reward):
        self.imps[item] += 1
        if reward > 0:
            self.clicks[item] += 1

    def get_clicks(self, user):
        return self.clicks

    def get_imps(self, user):
        return self.imps

    @property
    def label(self):
        return "CTR"


class CTRUserEstimator(Estimator):
    def reset(self, user_count, item_count):
        self.clicks = np.zeros((user_count, item_count))
        self.imps = np.zeros((user_count, item_count))

    def update(self, user, item, reward):
        self.imps[user, item] += 1
        if reward:
            self.clicks[user, item] += 1

    def get_clicks(self, user):
        return self.clicks[user]

    def get_imps(self, user):
        return self.imps[user]

    @property
    def label(self):
        return "uCTR"
