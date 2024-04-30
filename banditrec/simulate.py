import collections
import math
import random

from banditrec.agents import RandomAgent
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from banditrec.agent import ItemPool


class Simulator:
    def __init__(self, dataset):
        self.dataset = dataset

        self.time_horizon = dataset.time_horizon
        self.bucket_size = self.time_horizon // 500
        self.run_count = 1
        self.save_results = False

        self.tqdm_options = {}

        self.results = []

    def set_bucket_size(self, bucket_size):
        self.bucket_size = bucket_size
        return self

    def set_tqdm(self, options):
        self.tqdm_options = options
        return self

    def set_run_count(self, count):
        self.run_count = count
        return self

    def do_save_results(self, value):
        self.save_results = value
        return self

    @property
    def label(self):
        return "Simulator"

    @property
    def user_count(self):
        return self.dataset.interaction_matrix.shape[0]

    @property
    def item_count(self):
        return self.dataset.interaction_matrix.shape[1]

    def run(self, agent, horizon=None):
        rewards = np.zeros(math.ceil(self.time_horizon / self.bucket_size))

        t1 = (
            range(self.run_count)
            if self.run_count == 1
            else tqdm(range(self.run_count), postfix=agent.label, **self.tqdm_options)
        )

        for _ in t1:
            agent.reset(self.user_count, self.item_count)
            self.reset()

            # early stopping
            h = self.time_horizon if horizon is None else horizon
            t2 = (
                tqdm(range(h), postfix=agent.label, **self.tqdm_options)
                if self.run_count == 1
                else tqdm(range(h), leave=False)
            )

            for t in t2:
                user = self.get_user(t)
                pool = self.get_pool(t, user)

                item = agent.act(user, pool)
                reward = self.dataset.interaction_matrix[user, item]
                agent.update(user, item, reward)

                self.update(t, user, item, reward)
                rewards[t // self.bucket_size] += reward

        result = collections.namedtuple("SimResult", ["agent", "rewards"])(
            agent, rewards / self.run_count
        )
        if self.save_results:
            self.results.append(result)
        return result

    def reset(self):
        pass

    def get_user(self, t):
        return random.randrange(self.user_count)

    def get_pool(self, t, user):
        return ItemPool(np.ones(self.item_count, dtype=bool))

    def update(self, t, user, item, reward):
        pass

    def print_results(self):
        print(f"{self.label} | Final Agent CTR")
        for i, r in enumerate(self.results):
            ctr = r.rewards.mean() / self.bucket_size
            print(f"{i:>3}\t{ctr:.5f} - {r.agent.label}")

        print()

    def plot_cum_reward(self, relative=None, regret=False):
        fig1, ax = plt.subplots()

        for i, result in enumerate(self.results):
            values = (
                np.cumsum(self.bucket_size - result.rewards)
                if regret
                else np.cumsum(result.rewards)
            )
            if relative is not None:
                if i == relative:
                    continue
                values /= (
                    np.cumsum(self.bucket_size - self.results[relative].rewards)
                    if regret
                    else np.cumsum(self.results[relative].rewards)
                )

            x = self.bucket_size * np.arange(1, 1 + len(values))
            ax.plot(x, values, label=result.agent.label)

        prop = "regret" if regret else "reward"

        if relative is not None:
            plt.ylabel(f"Rel. cum. {prop}")
        else:
            plt.ylabel(f"Cum. {prop}")
        plt.xlabel("Time")
        plt.grid()
        plt.title(self.label)
        plt.xlim(left=0, right=x[-1])

        plt.legend()
        return plt

    def plot_cum_ctr(self):
        fig1, ax = plt.subplots()

        for result in self.results:
            values = np.cumsum(result.rewards)
            x = self.bucket_size * np.arange(1, 1 + len(values))

            ax.plot(x, values / x, label=result.agent.label)

        plt.ylabel("Cum. CTR")
        plt.xlabel("Time")
        plt.grid()
        plt.title(self.label)
        plt.xlim(left=0, right=x[-1])

        plt.legend()
        return plt


class SimulatorEC(Simulator):
    def reset(self):
        self.__pool_mask = np.ones((self.user_count, self.item_count), dtype=bool)

    def get_pool(self, t, user):
        return ItemPool(self.__pool_mask[user])

    def update(self, t, user, item, reward):
        self.__pool_mask[user, item] = False

    @property
    def label(self):
        return f"{self.dataset.name} EC@{self.time_horizon}"


class SimulatorRP(Simulator):
    def __init__(
        self,
        dataset,
        pool_size,
        allow_repeat_clicks=False,
    ):
        super().__init__(dataset)
        self.pool_size = pool_size
        self.allow_repeat_clicks = allow_repeat_clicks

    def reset(self):
        self.__user_imps = self.item_count - self.dataset.interaction_matrix.sum(axis=1)
        self.__pool_mask = ~self.dataset.interaction_matrix.astype(bool)
        self.__options = {
            u: list(self.dataset.interaction_matrix[u].nonzero()[0])
            for u in range(self.user_count)
        }
        for u in range(self.user_count):
            if len(self.__options[u]) == 0:
                del self.__options[u]

    def get_user(self, t):
        if self.allow_repeat_clicks:
            return random.randrange(self.user_count)
        else:
            return random.choice(list(self.__options.keys()))

    def get_pool(self, t, user):
        if self.allow_repeat_clicks:
            random_items = np.random.choice(
                self.item_count, self.pool_size - 1, replace=False
            )
        else:
            random_items = np.random.choice(
                np.arange(self.item_count),
                self.pool_size - 1,
                replace=False,
                p=np.divide(self.__pool_mask[user], self.__user_imps[user]),
            )

        clicked_item = random.choice(self.__options[user])

        pool = np.zeros(self.item_count, dtype=bool)
        pool[clicked_item] = True
        pool[random_items] = True
        return ItemPool(pool)

    def update(self, t, user, item, reward):
        if not self.allow_repeat_clicks:
            if reward:
                self.__options[user].remove(item)
                if len(self.__options[user]) == 0:
                    del self.__options[user]
            else:
                self.__pool_mask[user, item] = False
                self.__user_imps[user] -= 1

    @property
    def label(self):
        return f"{self.dataset.name} RP{self.pool_size}@{self.time_horizon}"


class SimulatorSW(Simulator):
    def __init__(
        self,
        dataset,
        pool_size,
        seed=0,
    ):
        super().__init__(dataset)
        self.pool_size = pool_size

        np.random.seed(seed)
        self.items = np.random.choice(
            np.arange(self.item_count), self.item_count, replace=False
        )
        np.random.seed()

        self.L = self.time_horizon / (self.item_count - self.pool_size + 1)

    def reset(self):
        self.__pool_masks = np.ones((self.user_count, self.item_count), dtype=bool)
        self.__pool_mask = np.zeros(self.item_count, dtype=bool)
        self.__pool_mask[self.items[: self.pool_size]] = True

    def get_pool(self, t, user):
        pool_min = int(t // self.L)
        pool_max = pool_min + self.pool_size
        if not self.__pool_mask[self.items[pool_max - 1]]:
            self.__pool_mask = np.zeros(self.item_count, dtype=bool)
            self.__pool_mask[self.items[pool_min:pool_max]] = True
        return ItemPool(self.__pool_masks[user] & self.__pool_mask)

    def update(self, t, user, item, reward):
        self.__pool_masks[user, item] = False

    @property
    def label(self):
        return f"{self.dataset.name} SW{self.pool_size}@{self.time_horizon}"


def simulate_replay(agent, events, pool_masks, T=None):
    user_count = events.user.nunique()
    item_count = pool_masks.shape[1]

    agent.reset(user_count, item_count)
    rewards = []

    if T is not None:
        e = events.iloc[:T]
    else:
        e = events

    for row in tqdm(e.itertuples(index=False), total=len(e)):
        pool = ItemPool(pool_masks[row.pool])
        user = row.user
        displayed_item = row.item

        item = agent.act(user, pool)

        if item != displayed_item:
            continue

        clicked = row.clicked
        agent.update(user, item, clicked)

        rewards.append(clicked)

    return collections.namedtuple("ReplayRes", ["agent", "rewards"])(
        agent, np.array(rewards)
    )
