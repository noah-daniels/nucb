import math
import numpy as np
import networkx as nx
import random


from banditrec.agent import Agent, EstimatorAgent
from banditrec.utils import IncrementalCosine2, IncrementalCosine3
from banditrec.estimators import CTREstimator


class RandomAgent(Agent):
    """
    At recommendation time, select a random item from the available items.
    """

    def act(self, user, pool):
        return random.choice(pool.items())

    @property
    def label(self):
        return "Random"


class HindsightAgent(Agent):
    """
    Rank the items based on their occurrence in the dataset.
    At recommendation time, select the highest ranked item from the available items.
    """

    def __init__(self, ranked_items=None):
        if ranked_items is not None:
            self.set_ranked_items(ranked_items)

    def set_ranked_items(self, ranked_items):
        self.ranked_items = ranked_items
        N = ranked_items.shape[0]
        self.scores = np.zeros(N)
        self.scores[self.ranked_items] = -np.arange(N)

    def act(self, user, pool):
        return pool.masked_argmax(self.scores)

    @property
    def label(self):
        return "Hindsight"


class EpsGreedyAgent(EstimatorAgent):
    def __init__(self, estimator, eps):
        super().__init__(estimator)
        self.eps = eps

    def act(self, user, pool):
        act_randomly = np.random.uniform(0, 1) < self.eps

        if act_randomly:
            return random.choice(pool.items())
        else:
            scores = self.estimator.get_ctrs(user)
            return pool.masked_argmax(scores)

    @property
    def label(self):
        return f"EG({self.estimator.label}, ε={self.eps})"


class UCBAgent(EstimatorAgent):
    def __init__(self, alpha, estimator=None, addone=False):
        if estimator is None:
            estimator = CTREstimator()
        super().__init__(estimator)
        self.alpha = alpha
        self.addone = addone

    def act(self, user, pool):
        imps = self.estimator.get_imps(user) + int(self.addone)
        ctrs = self.estimator.get_ctrs(user)

        n = self.t
        # n = imps.sum()
        scores = ctrs + self.alpha * np.sqrt(
            np.divide(
                np.log(n) if n >= 1 else np.inf,
                imps,
                where=imps > 0,
                out=np.zeros(imps.shape) + np.inf,
            )
        )

        return pool.masked_argmax(scores)

    @property
    def label(self):
        return (
            f"UCB{'*' if self.addone else ''}({self.estimator.label}, α={self.alpha})"
        )


class TSAgent(EstimatorAgent):
    def __init__(self, estimator, prior=None):
        super().__init__(estimator)
        self.prior = prior or [1, 1]

    def act(self, user, pool):
        clicks = self.estimator.get_clicks(user)
        imps = self.estimator.get_imps(user)

        scores = np.random.beta(self.prior[0] + clicks, self.prior[1] + imps - clicks)

        return pool.masked_argmax(scores)

    @property
    def label(self):
        if self.prior == [1, 1]:
            return f"TS({self.estimator.label})"
        else:
            return f"TS({self.estimator.label}, prior={self.prior})"


class LinearEstimator:
    def __init__(self, d, semicontextual=False):
        self.semicontextual = semicontextual

        if self.semicontextual:
            self.imps = np.ones(d)
            self.clicks = np.zeros(d)
            self.ctrs = np.zeros(d)
        else:
            self.M = np.identity(d)
            self.Minv = np.identity(d)
            self.b = np.zeros(d)
            self.w = np.zeros(d)

    def get_cb(self, x, t):
        if self.semicontextual:
            base = 1 / self.imps[x]
        else:
            base = x @ self.Minv @ x

        return math.sqrt(math.log(t + 1) * base)

    def get_value(self, x):
        if self.semicontextual:
            return self.ctrs[x]
        else:
            return np.dot(self.w, x)

    def update(self, x, reward):
        if self.semicontextual:
            self.clicks[x] += reward
            self.imps[x] += 1
            self.ctrs[x] = self.clicks[x] / self.imps[x]
        else:
            self.M += np.outer(x, x)
            self.b += reward * x
            self.Minv = np.linalg.inv(self.M)
            self.w = self.Minv @ self.b

    @property
    def coefficient(self):
        return self.ctrs if self.semicontextual else self.w


class LinUCBAgent(Agent):
    def __init__(self, alpha, personalized):
        super().__init__()
        self.alpha = alpha
        self.personalized = personalized

    def reset(self, user_count, item_count):
        assert self.item_features is not None

        d = self.item_features.shape[1]

        if self.personalized:
            self.estimators = [LinearEstimator(d) for _ in range(user_count)]
        else:
            self.estimator = LinearEstimator(d)

        self.t = 1

    def act(self, user, pool):
        available_items = pool.items()

        if self.personalized:
            estimator = self.estimators[user]
        else:
            estimator = self.estimator

        scores = np.empty(len(available_items))
        for i, item in enumerate(available_items):
            x = self.item_features[item]
            scores[i] = estimator.get_value(x) + self.alpha * estimator.get_cb(
                x, self.t
            )

        best_item_index = np.argmax(scores)
        return available_items[best_item_index]

    def update(self, user, item, reward):
        self.t += 1

        x = self.item_features[item]

        if self.personalized:
            estimator = self.estimators[user]
        else:
            estimator = self.estimator

        estimator.update(x, reward)

    @property
    def label(self):
        prefix = "IND" if self.personalized else "SIN"
        return f"{prefix}-LinUCB(α={self.alpha})"


class CLUBAgent(Agent):
    class CLUBCluster:
        def __init__(self, d, semicontextual=False):
            self.semicontextual = semicontextual
            self.estimator = LinearEstimator(d, semicontextual)

        def add_users(self, estimators):
            if self.semicontextual:
                clicks = self.estimator.clicks
                imps = self.estimator.imps
                for est in estimators:
                    imps += est.imps - np.ones(imps.shape)
                    clicks += est.clicks

                self.estimator.clicks = clicks
                self.estimator.imps = imps
                self.estimator.ctrs = self.estimator.clicks / self.estimator.imps
            else:
                d = self.estimator.M.shape[0]

                M = self.estimator.M
                b = self.estimator.b
                for est in estimators:
                    M += est.M - np.identity(d)
                    b += est.b

                self.estimator.M = M
                self.estimator.b = b
                self.estimator.Minv = np.linalg.inv(self.estimator.M)
                self.estimator.w = np.dot(self.estimator.Minv, self.estimator.b)

        def update(self, x, reward):
            self.estimator.update(x, reward)

        def get_scores(self, X, alpha, t):
            e = self.estimator
            return np.array([e.get_value(x) + alpha * e.get_cb(x, t) for x in X])

    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def reset(self, user_count, item_count):
        super().reset(user_count, item_count)

        if self.item_features is None:
            d = item_count
            self.estimators = [LinearEstimator(d, True) for _ in range(user_count)]
        else:
            d = self.item_features.shape[1]
            self.estimators = [LinearEstimator(d, False) for _ in range(user_count)]

        self.t = 1
        self.user_imps = np.zeros(user_count)

        # self.graph = nx.complete_graph(user_count)
        p = 3 * math.log(user_count) / user_count
        self.graph = nx.binomial_graph(user_count, p)

        self.cluster_map = dict()
        for c in nx.connected_components(self.graph):
            cluster = CLUBAgent.CLUBCluster(d, self.item_features is None)
            for u in c:
                self.cluster_map[u] = cluster

    def act(self, user, pool):
        # get item features
        available_items = pool.items()
        if self.item_features is None:
            X = available_items
        else:
            X = self.item_features[available_items]

        # determine scores of available items given parameters of cluster
        cluster = self.cluster_map[user]
        scores = cluster.get_scores(X, self.alpha, self.t)

        # select item with highest score
        best_item_index = np.argmax(scores)
        return available_items[best_item_index]

    def update(self, user, item, reward):
        self.t += 1

        x = item if self.item_features is None else self.item_features[item]

        # update user statistics
        self.estimators[user].update(x, reward)

        # update cluster statistics
        self.cluster_map[user].update(x, reward)

        # update clustering
        cb_i = self.beta * math.sqrt(
            (1 + math.log(1 + self.user_imps[user])) / (1 + self.user_imps[user])
        )
        w_i = self.estimators[user].coefficient

        to_delete = []
        for l in self.graph.neighbors(user):
            cb_l = self.beta * math.sqrt(
                (1 + math.log(1 + self.user_imps[l])) / (1 + self.user_imps[l])
            )
            w_l = self.estimators[l].coefficient
            diff = np.linalg.norm(w_i - w_l)
            if diff > cb_i + cb_l:
                to_delete.append(l)

        for l in to_delete:
            self.graph.remove_edge(user, l)
            # split cluster
            if not nx.has_path(self.graph, user, l):
                d = (
                    self.item_count
                    if self.item_features is None
                    else self.item_features.shape[1]
                )
                ic = list(nx.node_connected_component(self.graph, user))
                icluster = CLUBAgent.CLUBCluster(d, self.item_features is None)
                icluster.add_users([self.estimators[i] for i in ic])

                lc = list(nx.node_connected_component(self.graph, l))
                lcluster = CLUBAgent.CLUBCluster(d, self.item_features is None)
                lcluster.add_users([self.estimators[i] for i in lc])

                for u in ic:
                    self.cluster_map[u] = icluster
                for u in lc:
                    self.cluster_map[u] = lcluster

        self.user_imps[user] += 1

    @property
    def label(self):
        return f"CLUB(α={self.alpha}, b={self.beta})"


class DynUCBAgent(Agent):
    def __init__(self, alpha, cluster_count):
        super().__init__()
        self.alpha = alpha
        self.cluster_count = cluster_count

    def reset(self, user_count, item_count):
        super().reset(user_count, item_count)

        self.t = 1
        self.clusters = np.random.randint(self.cluster_count, size=user_count)

        if self.item_features is None:
            self.imps = np.ones((user_count, item_count))
            self.clicks = np.zeros((user_count, item_count))

            self.imps_cluster = np.ones((self.cluster_count, item_count))
            self.clicks_cluster = np.zeros((self.cluster_count, item_count))
            self.ctrs_cluster = np.zeros((self.cluster_count, item_count))
        else:
            d = self.item_features.shape[1]

            self.M = np.array([np.identity(d) for _ in range(user_count)])
            self.b = np.zeros((user_count, d))

            self.M_cluster = np.array(
                [np.identity(d) for _ in range(self.cluster_count)]
            )
            self.MInv_cluster = np.array(
                [np.identity(d) for _ in range(self.cluster_count)]
            )
            self.b_cluster = np.zeros((self.cluster_count, d))
            self.w_cluster = np.zeros((self.cluster_count, d))

    def act(self, user, pool):
        available_items = pool.items()
        cluster = self.clusters[user]

        if self.item_features is None:
            scores = self.ctrs_cluster[cluster, available_items] + self.alpha * np.sqrt(
                np.log(1 + self.t) / self.imps_cluster[cluster, available_items]
            )
        else:
            scores = np.empty(len(available_items))
            for i, item in enumerate(available_items):
                x = self.item_features[item]
                cb = self.alpha * np.sqrt(
                    x.T @ self.MInv_cluster[cluster] @ x * np.log(self.t + 1)
                )
                scores[i] = self.w_cluster[cluster].T @ x + cb

        best_item_index = np.argmax(scores)
        return available_items[best_item_index]

    def update(self, user, item, reward):
        self.t += 1

        current_cluster = self.clusters[user]

        # update statistics
        if self.item_features is None:
            self.imps[user, item] += 1
            self.clicks[user, item] += reward
            self.imps_cluster[current_cluster, item] += 1
            self.clicks_cluster[current_cluster, item] += reward
        else:
            x = self.item_features[item]
            dm = np.outer(x, x)
            self.M[user] += dm
            self.b[user] += reward * x
            self.M_cluster[current_cluster] += dm
            self.b_cluster[current_cluster] += reward * x

        # calculate potential new cluster
        if self.item_features is None:
            ctrs = self.clicks[user] / self.imps[user]
            distances = np.linalg.norm(ctrs - self.ctrs_cluster, axis=1)
        else:
            w = np.linalg.inv(self.M[user]) @ self.b[user]
            distances = np.linalg.norm(w - self.w_cluster, axis=1)
        new_cluster = np.argmin(distances)

        # move user to new cluster if necessary
        if new_cluster != current_cluster:
            previous_cluster = current_cluster
            self.clusters[user] = new_cluster

            if self.item_features is None:
                self.imps_cluster[new_cluster] += self.imps[user] - 1
                self.clicks_cluster[new_cluster] += self.clicks[user]
                self.imps_cluster[previous_cluster] -= self.imps[user] - 1
                self.clicks_cluster[previous_cluster] -= self.clicks[user]
                self.ctrs_cluster[new_cluster] = (
                    self.clicks_cluster[new_cluster] / self.imps_cluster[new_cluster]
                )
            else:
                d = self.item_features.shape[1]
                self.M_cluster[new_cluster] += self.M[user] - np.identity(d)
                self.b_cluster[new_cluster] += self.b[user]
                self.M_cluster[previous_cluster] -= self.M[user] - np.identity(d)
                self.b_cluster[previous_cluster] -= self.b[user]
                self.MInv_cluster[new_cluster] = np.linalg.inv(
                    self.M_cluster[new_cluster]
                )
                self.w_cluster[new_cluster] = (
                    self.MInv_cluster[new_cluster] @ self.b_cluster[new_cluster]
                )

        # update current cluster
        if self.item_features is None:
            self.ctrs_cluster[current_cluster, item] = (
                self.clicks_cluster[current_cluster, item]
                / self.imps_cluster[current_cluster, item]
            )
        else:
            self.MInv_cluster[current_cluster] = np.linalg.inv(
                self.M_cluster[current_cluster]
            )
            self.w_cluster[current_cluster] = (
                self.MInv_cluster[current_cluster] @ self.b_cluster[current_cluster]
            )

    @property
    def label(self):
        return f"DynUCB(α={self.alpha}, K={self.cluster_count})"


class CABAgent(Agent):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def reset(self, user_count, item_count):
        super().reset(user_count, item_count)

        self.t = 1

        if self.item_features is None:
            self.__reset_simlified()
        else:
            self.__reset_normal2()

    def act(self, user, pool):
        if self.item_features is None:
            return self.__act_simplified(user, pool)
        else:
            return self.__act_normal2(user, pool)

    def update(self, user, item, reward):
        self.t += 1

        if self.item_features is None:
            self.__update_simplified(user, item, reward)
        else:
            self.__update_normal2(user, item, reward)

    @property
    def label(self):
        return f"CAB(α={self.alpha}, γ={self.gamma})"

    def __reset_normal2(self):
        d = self.item_features.shape[1]

        self.b = np.zeros((self.user_count, d))
        self.M = np.array([np.identity(d) for _ in range(self.user_count)])
        self.Minv = np.linalg.inv(self.M)
        self.w = np.zeros((self.user_count, d))

        self.user_imps = np.zeros(self.user_count, dtype=bool)

    def __act_normal2(self, user, pool):
        available_items = pool.items()

        self.cb = np.empty((self.user_count, len(available_items)))
        scores = np.empty(len(available_items))
        for i, k in enumerate(available_items):
            x = self.item_features[k]

            cbb = x @ self.Minv @ x
            self.cb[:, i] = self.alpha * math.sqrt(math.log(self.t + 1)) * np.sqrt(cbb)

            a = np.dot(self.w[user], x) - np.dot(self.w, x)
            b = self.cb[user, i] + self.cb[:, i]
            mask = (np.abs(a) <= b) & self.user_imps
            mask[user] = True

            w_cluster = self.w[mask].sum(axis=0) / mask.sum()
            cb_cluster = self.cb[mask, i].sum() / mask.sum()

            scores[i] = np.dot(w_cluster, x) + cb_cluster

        self.idx = np.argmax(scores)
        return available_items[self.idx]

    def __update_normal2(self, user, item, reward):
        x = self.item_features[item]

        dm = np.outer(x, x)

        self.user_imps[user] = True

        if self.cb[user, self.idx] >= self.gamma / 4:
            self.M[user] += dm
            self.b[user] += reward * x
            self.Minv[user] = np.linalg.inv(self.M[user])
            self.w[user] = self.Minv[user] @ self.b[user]
        else:
            a = np.dot(self.w[user], x) - np.dot(self.w, x)
            b = self.cb[user, self.idx] + self.cb[:, self.idx]
            mask = (np.abs(a) <= b) & self.user_imps

            for j in mask.nonzero()[0]:
                if self.cb[j, self.idx] < self.gamma / 4:
                    self.M[j] += dm
                    self.b[j] += reward * x
                    self.Minv[j] = np.linalg.inv(self.M[j])
                    self.w[j] = self.Minv[j] @ self.b[j]

    def __reset_simlified(self):
        self.clicks = np.zeros((self.user_count, self.item_count))
        self.imps = np.ones((self.user_count, self.item_count))
        self.ctrs = self.clicks / self.imps
        self.cbs = self.alpha * np.sqrt(np.log(1 + self.t) / self.imps)

        self.user_imps = np.zeros(self.user_count, dtype=bool)

    def __act_simplified(self, user, pool):
        available_items = pool.items()

        scores = np.empty(len(available_items))
        for i, item in enumerate(available_items):
            mask = self.calculate_neighborhood_mask(user, item)
            scores[i] = (
                self.ctrs[mask, item].sum() + self.cbs[mask, item].sum()
            ) / mask.sum()

        best_idx = np.argmax(scores)
        return available_items[best_idx]

    def __update_simplified(self, user, item, reward):
        if self.cbs[user, item] >= self.gamma / 4:
            users = [user]
        else:
            mask = self.calculate_neighborhood_mask(user, item)
            users = mask.nonzero()[0]
            users = users[self.cbs[users, item] < (self.gamma / 4)]

        for u in users:
            self.imps[u, item] += 1
            self.clicks[u, item] += reward
            self.ctrs[u, item] = self.clicks[u, item] / self.imps[u, item]

        self.cbs = self.alpha * np.sqrt(np.log(1 + self.t) / self.imps)

    def calculate_neighborhood_mask(self, user, item):
        a = self.ctrs[user, item] - self.ctrs[:, item]
        b = self.cbs[user, item] + self.cbs[:, item]

        mask = (np.abs(a) <= b) & self.user_imps
        mask[user] = True
        return mask


class UserNeighborAgent(Agent):
    def __init__(self, prior=None):
        super().__init__()
        self.prior = prior or [1, 1]

    def reset(self, user_count, item_count):
        super().reset(user_count, item_count)
        self.clicks = np.zeros((user_count, item_count))

        self.mutual_clicks = np.zeros((user_count, user_count)) + self.prior[0]
        self.mutual_clicks[np.diag_indices(user_count)] += self.prior[1]

    def act(self, user, pool):
        mask = self.clicks[:, pool.items()].sum(axis=1)
        mask[user] = 0
        users = mask.nonzero()[0]

        if len(users) == 0:
            return pool.masked_argallmax(np.zeros(self.item_count))
        else:
            a = self.mutual_clicks[user][users].copy()
            b = self.mutual_clicks.diagonal()[users] - a
            scores = np.random.beta(a, b)
            u = users[np.argmax(scores)]

        return pool.masked_argallmax(self.clicks[u])

    def update(self, user, item, reward):
        if reward:
            self.clicks[user, item] += 1
            self.mutual_clicks[user, :] += self.clicks[:, item] > 0
            self.mutual_clicks[:, user] += self.clicks[:, item] > 0
            self.mutual_clicks[user, user] -= 1

    @property
    def label(self):
        if self.prior == [1, 1]:
            return f"UN"
        else:
            return f"UN({self.prior})"


class SCLUBAgent(Agent):
    class Cluster:
        def __init__(self, agent):
            self.agent = agent
            self.T = 0
            self.users = []

            if self.contextual:
                self.M = np.identity(self.d)
                self.b = np.zeros(self.d)
                self.Minv = np.linalg.inv(self.M)
                self.w = np.dot(self.Minv, self.b)
            else:
                self.imps = np.ones(self.d)
                self.clicks = np.zeros(self.d)
                self.w = self.clicks / self.imps

        @property
        def contextual(self):
            return self.agent.item_features is not None

        @property
        def d(self):
            return (
                self.agent.item_features.shape[1]
                if self.contextual
                else self.agent.item_count
            )

        def get_scores(self, available_items, t, beta):
            if self.contextual:
                X = self.agent.item_features[available_items]
                mean = np.array([np.dot(x, self.w) for x in X])
                cb = np.sqrt(np.array([x @ self.Minv @ x for x in X]))
            else:
                mean = self.w[available_items]
                cb = np.sqrt(1 / self.imps[available_items])

            return mean + beta * math.sqrt(math.log(t + 1)) * cb

        def update(self, item, reward):
            self.T += 1

            if self.contextual:
                x = self.agent.item_features[item]
                self.M += np.outer(x, x)
                self.b += reward * x
            else:
                self.imps[item] += 1
                self.clicks[item] += reward

            self.__recalculate_w(item)

        def remove_user(self, user):
            self.T -= user.T
            self.users.remove(user.user_index)

            if self.contextual:
                self.M -= user.M - np.identity(self.d)
                self.b -= user.b
            else:
                self.imps -= user.imps - np.ones(self.d)
                self.clicks -= user.clicks

            self.__recalculate_w()

        def add_user(self, user):
            self.T += user.T
            self.users.append(user.user_index)

            if self.contextual:
                self.M += user.M - np.identity(self.d)
                self.b += user.b
            else:
                self.imps += user.imps - np.ones(self.d)
                self.clicks += user.clicks

            self.__recalculate_w()

        def merge(self, other):
            self.T += other.T
            self.users += other.users

            if self.contextual:
                self.M += other.M - np.identity(self.d)
                self.b += other.b
            else:
                self.imps += other.imps - np.ones(self.d)
                self.clicks += other.clicks

            self.__recalculate_w()

        def __recalculate_w(self, item=None):
            if self.contextual:
                self.Minv = np.linalg.inv(self.M)
                self.w = self.Minv @ self.b
            else:
                if item is None:
                    self.w = self.clicks / self.imps
                else:
                    self.w[item] = self.clicks[item] / self.imps[item]

        def is_checked(self, checked_users):
            return checked_users[self.users].sum() == len(self.users)

    class User:
        def __init__(self, user_index, agent):
            self.agent = agent
            self.user_index = user_index
            self.T = 0

            if self.agent.item_features is None:
                d = self.agent.item_count
                self.user_index = user_index
                self.imps = np.ones(d)
                self.clicks = np.zeros(d)
                self.w = self.clicks / self.imps
            else:
                d = self.agent.item_features.shape[1]
                self.M = np.identity(d)
                self.b = np.zeros(d)
                self.Minv = np.identity(d)
                self.w = np.zeros(d)

        def update(self, item, reward):
            self.T += 1

            if self.agent.item_features is None:
                self.imps[item] += 1
                self.clicks[item] += reward
                self.w[item] = self.clicks[item] / self.imps[item]
            else:
                x = self.agent.item_features[item]
                self.M += np.outer(x, x)
                self.b += reward * x
                self.Minv = np.linalg.inv(self.M)
                self.w = self.Minv @ self.b

    def __init__(self, beta, alpha1, alpha2):
        super().__init__()
        self.beta = beta
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def reset(self, user_count, item_count):
        super().reset(user_count, item_count)

        # user information
        self.users = [SCLUBAgent.User(u, self) for u in range(self.user_count)]
        self.user_T = np.zeros(self.user_count)

        # cluster information
        self.clusters = {
            0: SCLUBAgent.Cluster(self),
        }
        self.cluster_map = np.zeros(self.user_count, dtype=int)
        self.clusters[0].users = list(range(self.user_count))

        # time and phase control
        self.t = 0
        self.c = 0
        self.__advance_time()

    def act(self, user, pool):
        # get cluster for current user
        cluster_index = self.cluster_map[user]
        cluster = self.clusters[cluster_index]

        # calcuclate scores for available items
        available_items = pool.items()
        scores = cluster.get_scores(available_items, self.t, self.beta)

        # return item with highest score
        best_item_index = np.argmax(scores)
        return available_items[best_item_index]

    def update(self, user, item, reward):
        # update user and cluster statistics
        self.__update(user, item, reward)

        # potentially split cluster
        self.__split(user)

        # mark user checked
        self.__mark(user)

        # potentially merge clusters
        self.__merge(user)

        # increase time (and advance phase if necessary)
        self.__advance_time()

    @property
    def label(self):
        return f"SCLUB(β={self.beta}, α1={self.alpha1}, α2={self.alpha2})"

    def __update(self, user_index, item, reward):
        # update user
        user = self.users[user_index]
        user.update(item, reward)
        self.user_T[user_index] += 1

        # update cluster
        cluster_index = self.cluster_map[user_index]
        cluster = self.clusters[cluster_index]
        cluster.update(item, reward)

    def __split(self, user_index):
        user = self.users[user_index]

        cluster_index = self.cluster_map[user_index]
        cluster = self.clusters[cluster_index]
        pivot_T, pivot_w = self.pivots[cluster_index]

        p = self.user_T[user_index] / self.t

        do_split = False
        if np.linalg.norm(user.w - pivot_w) > self.alpha1 * (
            self.__F(user.T) + self.__F(pivot_T)
        ):
            do_split = True
        else:
            pis = self.user_T[cluster.users] / self.t
            if np.any(np.abs(p - pis) > 2 * self.alpha2 * self.__F(self.t)):
                do_split = True

        if do_split and len(cluster.users) > 1:
            new_cluster_index = max(self.clusters.keys()) + 1
            new_cluster = SCLUBAgent.Cluster(self)
            self.clusters[new_cluster_index] = new_cluster
            self.cluster_map[user_index] = new_cluster_index

            new_cluster.add_user(user)
            self.pivots[new_cluster_index] = (new_cluster.T, new_cluster.w)

            cluster.remove_user(user)

    def __mark(self, user):
        self.checked_users[user] = True
        cluster_index = self.cluster_map[user]
        cluster = self.clusters[cluster_index]
        if cluster.is_checked(self.checked_users):
            self.checked_clusters.add(cluster_index)

    def __merge(self, user_index):
        j1 = self.cluster_map[user_index]
        if j1 not in self.checked_clusters:
            return

        c1 = self.clusters[j1]
        p1 = c1.T / (len(c1.users) * self.t)

        to_merge = []
        for j2 in self.checked_clusters:
            if j1 == j2:
                continue

            c2 = self.clusters[j2]
            p2 = c2.T / (len(c2.users) * self.t)

            cond1 = np.linalg.norm(c1.w - c2.w) < 0.5 * self.alpha1 * (
                self.__F(c1.T) + self.__F(c2.T)
            )
            cond2 = abs(p1 - p2) < self.alpha2 * self.__F(self.t)

            if cond1 and cond2:
                to_merge.append(j2)

        for j2 in to_merge:
            c2 = self.clusters[j2]
            c1.merge(c2)
            del self.clusters[j2]
            self.checked_clusters.remove(j2)
            self.cluster_map[self.cluster_map == j2] = j1

    def __advance_time(self):
        self.t += 1

        # new phase
        if math.log2(self.t + 1) == self.c + 1:
            self.c += 1
            self.checked_users = np.zeros(self.user_count, dtype=bool)
            self.checked_clusters = set()
            self.pivots = {j: (c.T, c.w) for j, c in self.clusters.items()}

    def __F(self, t):
        a = 1 + math.log(1 + t)
        b = 1 + t
        return math.sqrt(a / b)


class NLinUCBAgent(Agent):
    def __init__(self, alpha, uw):
        super().__init__()

        self.alpha = alpha
        self.beta = uw

        self.use_global_info = True
        self.use_normalization = True
        self.use_exact_version = False
        self.similarity_type = "b"
        self.user_limit_count = None
        self.user_limit_method = None

    def toggle_global_info(self, value=None):
        self.use_global_info = not self.use_global_info if value is None else value
        return self

    def toggle_normalization(self, value=None):
        self.use_normalization = not self.use_normalization if value is None else value
        return self

    def toggle_exact_version(self, value=None):
        self.use_exact_version = not self.use_exact_version if value is None else value
        return self

    def set_similarity_type(self, similarity_type):
        assert similarity_type in ["w", "b"]
        self.similarity_type = similarity_type
        return self

    def set_user_limit(self, count, method=None):
        assert count is None or method in [
            "click_window",
            "imp_window",
            "random",
        ]
        self.user_limit_count = count
        self.user_limit_method = method
        return self

    @property
    def use_window(self):
        return self.user_limit_count is not None and self.user_limit_method in [
            "click_window",
            "imp_window",
        ]

    @property
    def d(self):
        return self.item_features.shape[1]

    def reset(self, user_count, item_count):
        assert self.item_features is not None
        super().reset(user_count, item_count)

        self.t = 1

        # global estimator
        self.M = np.zeros((self.d, self.d))
        self.b = np.zeros(self.d)

        # personal estimators
        self.M_user = np.zeros((self.user_count, self.d, self.d))
        self.b_user = np.zeros((self.user_count, self.d))

        # precomputed aggregate estimators
        if not self.use_exact_version:
            self.M_user2 = np.zeros((self.user_count, self.d, self.d))
            self.b_user2 = np.zeros((self.user_count, self.d))

        # user window
        if self.use_window:
            self.user_queue = []

        # similarity measure
        self.similarity_users = IncrementalCosine3(user_count, self.d)

    def act(self, user, pool):
        available_items = pool.items()

        # determine which other users to consider
        if self.user_limit_count is None:
            other_users = np.arange(self.user_count)
        else:
            if self.use_window:
                other_users = self.user_queue
            elif self.user_limit_method == "random":
                other_users = np.random.choice(
                    np.arange(self.user_count),
                    replace=False,
                    size=self.user_limit_count,
                )

        # aggregate estimators
        if self.use_exact_version:
            weights = self.beta * np.maximum(
                0, self.similarity_users.sims[user, other_users]
            )

            # calculate weighted sum
            M = np.identity(self.d) + np.matmul(
                weights, self.M_user[other_users].transpose(1, 0, 2)
            )
            b = np.dot(weights, self.b_user[other_users])
        else:
            M = np.identity(self.d) + self.M_user2[user]
            b = self.b_user2[user].copy()

        # add global information
        if self.use_global_info:
            M += self.M
            b += self.b

        # normalization
        if self.use_normalization:
            normalization_factor = np.linalg.norm(M) / np.linalg.norm(
                self.M + np.identity(self.d)
            )
        else:
            normalization_factor = 1

        # calculate scores
        Minv = np.linalg.inv(M)
        w = Minv @ b
        scores = np.empty(len(available_items))
        for i, item in enumerate(available_items):
            x = self.item_features[item]
            ctr = w.T @ x
            cb = self.alpha * np.sqrt(
                x.T @ Minv @ x * np.log(self.t + 1) * normalization_factor
            )
            scores[i] = ctr + cb

        # return best item
        best_item_index = np.argmax(scores)
        return available_items[best_item_index]

    def update(self, user, item, reward):
        self.t += 1

        # update global and personal estimators
        x = self.item_features[item]
        deltaM = np.outer(x, x)
        deltab = reward * x
        self.M += deltaM
        self.b += deltab
        self.M_user[user] += deltaM
        self.b_user[user] += deltab

        # update similarities
        if self.similarity_type == "w":
            v = (
                np.linalg.inv(np.identity(self.d) + self.M_user[user])
                @ self.b_user[user]
            )
        else:
            v = self.b_user[user]
        self.similarity_users.set_user(user, v)

        # update precomputed aggregated estimators
        if not self.use_exact_version:
            scores = self.beta * np.maximum(0, self.similarity_users.sims[user, :])
            self.M_user2 += scores[:, np.newaxis, np.newaxis] * deltaM
            self.b_user2 += scores[:, np.newaxis] * deltab

        # user window
        if self.use_window:
            if self.user_limit_method == "imp_window":
                self.user_queue.append(user)
            elif self.user_limit_method == "click_window" and reward:
                self.user_queue.append(user)

            if len(self.user_queue) > self.user_limit_count:
                self.user_queue.pop(0)

    @property
    def label(self):
        return f"NLinUCB(α={self.alpha}, β={self.beta})"


class NUCBAgent(Agent):
    def __init__(self, alpha, uw, iw):
        super().__init__()

        self.alpha = alpha
        self.beta1 = uw
        self.beta2 = iw

        self.use_global_info = True
        self.use_normalization = True
        self.use_exact_version = False
        self.use_incremental_cosine = True
        self.similarity_type = "c"
        self.user_limit_count = None
        self.user_limit_method = None

    def toggle_global_info(self, value=None):
        self.use_global_info = (not self.use_global_info) if value is None else value
        return self

    def toggle_normalization(self, value=None):
        self.use_normalization = (
            (not self.use_normalization) if value is None else value
        )
        return self

    def toggle_exact_version(self, value=None):
        self.use_exact_version = (
            (not self.use_exact_version) if value is None else value
        )
        return self

    def toggle_incremental_cosine(self, value=None):
        self.use_incremental_cosine = (
            not self.use_incremental_cosine if value is None else value
        )
        return self

    def set_similarity_type(self, similarity_type):
        assert similarity_type in ["c", "r", "logc"]
        self.similarity_type = similarity_type
        return self

    def set_user_limit(self, count, method=None):
        assert count is None or method in [
            "click_window",
            "imp_window",
            "random",
        ]
        self.user_limit_count = count
        self.user_limit_method = method
        return self

    @property
    def use_window(self):
        return self.user_limit_count is not None and self.user_limit_method in [
            "click_window",
            "imp_window",
        ]

    def reset(self, user_count, item_count):
        super().reset(user_count, item_count)

        self.t = 1

        # global statistics
        self.clicks_global = np.zeros(item_count)
        self.imps_global = np.zeros(item_count)

        # personal statistics
        self.clicks_user = np.zeros((user_count, item_count))
        self.imps_user = np.zeros((user_count, item_count))

        # precomputed aggregate statistics
        if not self.use_exact_version:
            self.clicks_precomputed = np.zeros((user_count, item_count))
            self.imps_precomputed = np.zeros((user_count, item_count))

        # user window
        if self.use_window:
            self.user_queue = []

        # similarity measure
        if self.use_incremental_cosine:
            if self.beta1 > 0:
                self.similarity_users = IncrementalCosine2(user_count, item_count)
            if self.beta2 > 0:
                self.similarity_items = IncrementalCosine2(item_count, user_count)

    def act(self, user, pool):
        available_items = pool.items()

        # determine which other users to consider
        if self.user_limit_count is None:
            other_users = np.arange(self.user_count)
        else:
            if self.use_window:
                other_users = self.user_queue
            elif self.user_limit_method == "random":
                other_users = np.random.choice(
                    np.arange(self.user_count),
                    replace=False,
                    size=self.user_limit_count,
                )

        # aggregate statistics
        if self.use_exact_version:
            sims = self.__get_user_sims(user, other_users)
            weights = self.beta1 * np.maximum(0, sims)
            mesh = np.ix_(other_users, available_items)
            clicks = np.dot(weights, self.clicks_user[mesh])
            imps = np.dot(weights, self.imps_user[mesh])
        else:
            clicks = self.clicks_precomputed[user, available_items]
            imps = self.imps_precomputed[user, available_items]

        # add global information
        if self.use_global_info:
            clicks += self.clicks_global[available_items]
            imps += self.imps_global[available_items]

        # normalization
        t = self.imps_global[available_items].sum()
        n = imps.sum()
        if self.use_normalization:
            normalization_factor = t / n if n > 0 else 0
        else:
            normalization_factor = 1
        clicks *= normalization_factor
        imps *= normalization_factor

        # calculate scores
        ctrs = np.divide(clicks, imps, where=imps > 0, out=np.zeros(imps.shape))
        cbs = self.alpha * np.sqrt(
            np.divide(
                math.log(t + 1),
                imps,
                where=imps > 0,
                out=np.zeros(imps.shape) + np.inf,
            )
        )
        scores = ctrs + cbs

        # return best item
        best_item_index = np.argmax(scores)
        return available_items[best_item_index]

    def update(self, user, item, reward):
        self.t += 1

        # update global and personal statistics
        self.imps_global[item] += 1
        self.imps_user[user, item] += 1
        self.clicks_global[item] += reward
        self.clicks_user[user, item] += reward

        # update similarities
        if self.use_incremental_cosine:
            if self.similarity_type == "c":
                v = self.clicks_user[user, item]
            elif self.similarity_type == "logc":
                v = math.log(1 + self.clicks_user[user, item])
            else:
                user_ctr = self.clicks_user[user].sum() / self.imps_user[user].sum()
                v = self.clicks_user[user, item] - user_ctr * (
                    self.imps_user[user, item] - self.clicks_user[user, item]
                )
            if self.beta1 > 0:
                self.similarity_users.set(user, item, v)
            if self.beta2 > 0:
                self.similarity_items.set(item, user, v)

        # update precomputed aggregate statistics
        if not self.use_exact_version:
            if self.beta1 > 0:
                sims = self.__get_user_sims(user, np.arange(self.user_count))
                scores = self.beta1 * np.maximum(0, sims)
                self.imps_precomputed[:, item] += scores
                self.clicks_precomputed[:, item] += reward * scores
            if self.beta2 > 0:
                sims = self.__get_item_sims(item, np.arange(self.item_count))
                scores = self.beta2 * np.maximum(0, sims)
                self.imps_precomputed[user, :] += scores
                self.clicks_precomputed[user, :] += reward * scores

        # user window
        if self.use_window:
            if self.user_limit_method == "imp_window":
                self.user_queue.append(user)
            elif self.user_limit_method == "click_window" and reward:
                self.user_queue.append(user)

            if len(self.user_queue) > self.user_limit_count:
                self.user_queue.pop(0)

    def __get_user_sims(self, user, users):
        if self.use_incremental_cosine:
            return self.similarity_users.sims[user, users]
        else:
            u = self.clicks_user[user]
            R = self.clicks_user[users]

            products = R @ u
            norms = np.linalg.norm(u) * np.linalg.norm(R, axis=1)
            sims = np.divide(
                products, norms, where=norms > 0, out=np.zeros(norms.shape)
            )
            return sims

    def __get_item_sims(self, item, items):
        if self.use_incremental_cosine:
            return self.similarity_items.sims[item, items]
        else:
            u = self.clicks_user[:, item]
            R = self.clicks_user[:, items]

            products = R @ u
            norms = np.linalg.norm(u) * np.linalg.norm(R, axis=1)
            sims = np.divide(
                products, norms, where=norms > 0, out=np.zeros(norms.shape)
            )
            return sims

    @property
    def label(self):
        return f"NUCB(α={self.alpha}, β1={self.beta1}, β2={self.beta2})"
