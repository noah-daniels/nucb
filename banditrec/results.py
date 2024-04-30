import matplotlib.pyplot as plt
import numpy as np


def plot_time_results(results, title):
    fig1, ax = plt.subplots()

    for res in results:
        values = np.divide(
            res.clicks, res.imps, where=res.imps > 0, out=np.zeros(res.imps.shape)
        )
        ax.plot(values, label=res.agent.label)

    plt.ylabel("CTR")
    plt.xlabel("Bucket Index")
    plt.xticks(np.arange(len(res.clicks)))
    plt.title(title)
    plt.grid()

    plt.legend()
    plt.show()


def plot_replay_results(results, title, start, end):
    fig1, ax = plt.subplots()

    for res in results:
        ctrs = np.cumsum(res.rewards) / np.cumsum(np.ones(res.rewards.shape))
        ctrs = ctrs[start:end]
        ax.plot(start + np.arange(0, len(ctrs)), ctrs, label=res.agent.label)

    plt.ylabel("CTR")
    plt.xlabel("Rounds")
    plt.title(title)
    plt.grid()

    plt.legend()
    plt.show()


def plot_cum_reward(
    results, title, bucket_size, relative_to=None, regret=False, ylim=None
):
    fig1, ax = plt.subplots()

    for result in results:
        values = (
            np.cumsum(bucket_size - result.rewards)
            if regret
            else np.cumsum(result.rewards)
        )
        if relative_to is not None:
            values /= (
                np.cumsum(bucket_size - relative_to.rewards)
                if regret
                else np.cumsum(relative_to.rewards)
            )

        x = bucket_size * np.arange(1, 1 + len(values))
        ax.plot(x, values, label=result.agent.label)

    plt.ylabel("Cum. reward")
    plt.xlabel("Time")
    plt.grid()
    plt.title(title)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlim(left=0, right=x[-1])

    plt.legend()
    plt.show()


def plot_cum_ctr(results, title, bucket_size, ylim=None):
    fig1, ax = plt.subplots()

    for result in results:
        values = np.cumsum(result.rewards)
        x = bucket_size * np.arange(1, 1 + len(values))

        ax.plot(x, values / x, label=result.agent.label)

    plt.ylabel("Cum. CTR")
    plt.xlabel("Time")
    plt.grid()
    plt.title(title)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlim(left=0, right=x[-1])

    plt.legend()
    plt.show()


def plot_ctr(results, title, bucket_size, ylim=None):
    fig1, ax = plt.subplots()

    miny = 1
    maxy = 0
    for res in results:
        values = res.rewards / bucket_size
        x = bucket_size * np.arange(1, len(values) + 1)
        ax.plot(x, values, label=res.agent.label)
        maxy = max(maxy, values[-1])
        miny = min(miny, values[-1])

    if ylim is not None:
        plt.ylim(ylim)

    plt.grid()

    plt.ylabel("CTR")
    plt.xlabel("Time")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_base_lifted_ctr(result, ground_ctrs, count, title):
    fig1, ax = plt.subplots()

    imps = result.item_imps
    clicks = result.item_clicks
    agent = result.agent
    item_count = imps.shape[0]

    idx = np.argsort(-imps)[:count]
    lifted = np.divide(clicks, imps, where=imps > 0, out=np.zeros(item_count))[idx]
    ground = ground_ctrs[idx]

    ax.scatter(ground, lifted, alpha=np.linspace(1, 0.1, count))

    xl, xr = ax.get_xlim()
    yl, yr = ax.get_ylim()
    l, r = min(xl, yl), max(xr, yr)
    ax.set_xlim(l, r)
    ax.set_ylim(l, r)
    ax.set_box_aspect(1)
    ax.set_xlabel("Ground CTR")
    ax.set_ylabel("Lifted CTR")
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="black", alpha=0.5)
    plt.title(agent.label)
    plt.suptitle(title)

    plt.show()
