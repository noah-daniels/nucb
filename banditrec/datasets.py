from collections import Counter, defaultdict
import fileinput
import re
import math

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_array


class InteractionDataset:
    def __init__(self, interaction_matrix, name, time_horizon):
        self.interaction_matrix = interaction_matrix
        self.name = name
        self.time_horizon = time_horizon
        self.item_features = None

    def add_item_features(self, item_features):
        self.item_features = item_features
        return self

    @property
    def ranked_items(self):
        return np.argsort(-self.interaction_matrix.sum(axis=0))


def parse_yahoo2(filenames):
    events = []

    users = dict()
    users_info = []

    items = dict()

    pools = dict()
    pools_info = []

    def parse_line(line):
        cols = line.split()

        # parse date, selected item and click result
        timestamp = int(cols[0])
        selected_item = int(cols[1][3:])
        clicked = bool(int(cols[2]))

        # parse user and items
        remainder = " ".join(cols[3:])[1:].split("|")
        user_features = tuple(
            map(lambda x: int(x) - 1, remainder[0].strip().split(" ")[1:])
        )
        available_items = tuple(int(i.strip()[3:]) for i in remainder[1:])

        return timestamp, selected_item, clicked, user_features, available_items

    with fileinput.input(files=filenames) as f:
        for line in tqdm(f):
            _, selected_item, clicked, user_features, available_items = parse_line(line)

            if user_features not in users:
                users[user_features] = len(users_info)
                users_info.append(user_features)
            user_id = users[user_features]

            if selected_item not in items:
                items[selected_item] = len(items.keys())
            selected_item_id = items[selected_item]

            available_item_ids = []
            for i in available_items:
                if i not in items:
                    items[i] = len(items.keys())
                available_item_ids.append(items[i])
            available_item_ids = tuple(available_item_ids)

            if available_item_ids not in pools:
                pools[available_item_ids] = len(pools_info)
                pools_info.append(available_item_ids)
            pool_id = pools[available_item_ids]

            events.append(
                [
                    selected_item_id,
                    clicked,
                    user_id,
                    pool_id,
                ]
            )

    df = pd.DataFrame(events, columns=["item", "clicked", "user", "pool"])

    user_features = np.zeros((len(users_info), 136))
    for user_id, nonzero in enumerate(users_info):
        user_features[user_id, nonzero] = 1

    pool_mask = np.zeros((len(pools_info), len(items.keys())))
    for pool_id, available in enumerate(pools_info):
        pool_mask[pool_id, available] = 1

    return df, user_features, pool_mask


def import_movielens(path, sep="::"):
    columns = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_table(path, sep=sep, header=None, names=columns, engine="python")
    R = (
        df.pivot(index="user_id", columns="item_id", values="rating")
        .fillna(0)
        .to_numpy()
    )
    return R, df


def import_lastfm(path, K=None):
    df = pd.read_csv(path, delimiter="\t")

    # keep top-K
    if K is not None:
        keep_items = df.artistID.value_counts().head(K).index
        result = df[df.artistID.isin(keep_items)].reset_index(drop=True)
    else:
        result = df

    # factorize users
    result["user_id"] = result.userID.factorize()[0]

    # randomize + factorize items
    ids = result.artistID.unique()
    np.random.shuffle(ids)
    labels, uniques = pd.factorize(ids)
    mapper = np.vectorize(dict(zip(uniques, labels)).get)
    result["item_id"] = mapper(result.artistID)

    result["clicked"] = True

    R = (
        result.pivot(index="user_id", columns="item_id", values="clicked")
        .fillna(False)
        .to_numpy()
    )

    return R, result


def import_delicious(base_path, K=None):
    df_user_bookmarks = pd.read_csv(
        f"{base_path}/user_taggedbookmarks-timestamps.dat",
        delimiter="\t",
        encoding="latin-1",
    )
    df_bookmarks = pd.read_csv(
        f"{base_path}/bookmarks.dat", delimiter="\t", encoding="latin-1"
    )
    joined = pd.merge(
        df_user_bookmarks, df_bookmarks, left_on="bookmarkID", right_on="id"
    ).drop_duplicates(["userID", "md5Principal"])[
        ["userID", "md5Principal", "urlPrincipal"]
    ]

    # keep top-k items
    if K is not None:
        keep_items = joined.md5Principal.value_counts().head(K).index
        result = joined[joined.md5Principal.isin(keep_items)].reset_index(drop=True)
    else:
        result = joined

    # factorize users
    result["user_id"] = result.userID.factorize()[0]
    result.drop(columns="userID", inplace=True)

    # randomize + factorize items
    items = result.md5Principal.unique()
    np.random.shuffle(items)
    labels, uniques = pd.factorize(items)
    mapper = np.vectorize(dict(zip(uniques, labels)).get)
    result["item_id"] = mapper(result.md5Principal)
    result.drop(columns="md5Principal", inplace=True)

    # create interaction matrix
    result["clicked"] = True
    R = (
        result.pivot(index="user_id", columns="item_id", values="clicked")
        .fillna(False)
        .to_numpy()
    )

    return R, result


def import_lastfm2(base_path, d=25, K=None):
    # load dataframes
    df_user_artists = pd.read_csv(
        f"{base_path}/user_artists.dat", delimiter="\t", encoding="latin-1"
    )
    df_tags = pd.read_csv(f"{base_path}/tags.dat", delimiter="\t", encoding="latin-1")
    df_artist_tags = pd.read_csv(
        f"{base_path}/user_taggedartists-timestamps.dat",
        delimiter="\t",
        encoding="latin-1",
    )

    # keep top-K
    if K is not None:
        keep_items = df_user_artists.artistID.value_counts().head(K).index
        df_user_artists = df_user_artists[
            df_user_artists.artistID.isin(keep_items)
        ].reset_index(drop=True)

    # randomize item IDs
    ids = df_user_artists.artistID.unique()
    np.random.shuffle(ids)
    labels, uniques = pd.factorize(ids)
    mapper = np.vectorize(dict(zip(uniques, labels)).get)
    df_user_artists["item_id"] = mapper(df_user_artists.artistID)
    active_artists = df_user_artists.artistID.unique()
    df_artist_tags = df_artist_tags.copy()[df_artist_tags.artistID.isin(active_artists)]
    df_artist_tags["item_id"] = mapper(df_artist_tags.artistID)

    # factorize user IDs
    df_user_artists["user_id"] = df_user_artists.userID.factorize()[0]

    def clean(x0):
        p1 = re.compile("[^A-Za-z0-9_\-' ]")
        p2 = re.compile("_|-|'")
        x1 = x0.lower().strip()
        x2 = re.sub(p1, "", x1)
        x3 = re.sub(p2, " ", x2)
        return x3

    df_tags["cleaned_tags"] = df_tags.tagValue.apply(clean)
    b = (
        pd.merge(df_artist_tags, df_tags, how="left")
        .groupby("item_id")
        .cleaned_tags.apply(list)
        .apply(lambda x: " ".join(map(str, x)))
    )

    vectorizer = TfidfVectorizer(strip_accents="unicode")
    X = vectorizer.fit_transform(b.values)
    pca = TruncatedSVD(n_components=d)
    X2 = pca.fit_transform(X)
    feature_vectors = np.zeros((df_user_artists.item_id.nunique(), 25))
    feature_vectors[b.index.values] = X2

    df_user_artists["clicked"] = True

    R = (
        df_user_artists.pivot(index="user_id", columns="item_id", values="clicked")
        .fillna(False)
        .to_numpy()
    )

    return R, feature_vectors


def import_delicious2(base_path, d=25, K=None):
    df_user_bookmarks = pd.read_csv(
        f"{base_path}/user_taggedbookmarks-timestamps.dat",
        delimiter="\t",
        encoding="latin-1",
    )
    df_tags = pd.read_csv(f"{base_path}/tags.dat", delimiter="\t", encoding="latin-1")

    # keep top-k items
    if K is not None:
        keep_items = df_user_bookmarks.bookmarkID.value_counts().head(K).index
        df_user_bookmarks = df_user_bookmarks[
            df_user_bookmarks.bookmarkID.isin(keep_items)
        ].reset_index(drop=True)

    # clean tags
    def clean_str(x):
        x = x.lower()
        x = re.sub("_|-", " ", x)
        x = re.sub("[^A-Za-z0-9\s]+", "", x)
        return x

    df_tags["value"] = df_tags.value.map(clean_str, na_action="ignore")

    # factorize userIDs
    df_user_bookmarks["user_id"] = df_user_bookmarks.userID.factorize()[0]

    # randomize + factorize bookmarkIDs
    items = df_user_bookmarks.bookmarkID.unique()
    np.random.shuffle(items)
    labels, uniques = pd.factorize(items)
    mapper = np.vectorize(dict(zip(uniques, labels)).get)
    df_user_bookmarks["item_id"] = mapper(df_user_bookmarks.bookmarkID)

    # # get tfidf vectors
    b = (
        pd.merge(df_user_bookmarks, df_tags, how="left", left_on="tagID", right_on="id")
        .groupby("item_id")
        .value.apply(list)
        .apply(lambda x: " ".join(map(str, x)))
    )
    vectorizer = TfidfVectorizer(strip_accents="unicode")
    raw_features = vectorizer.fit_transform(b.values)

    # PCA
    pca = TruncatedSVD(n_components=d)
    X = pca.fit_transform(raw_features)

    # get interaction matrix
    df_user_bookmarks = df_user_bookmarks[["user_id", "item_id", "tagID"]]
    df_user_bookmarks.drop_duplicates(["user_id", "item_id"], inplace=True)
    df_user_bookmarks["clicked"] = True
    R = (
        df_user_bookmarks.pivot(index="user_id", columns="item_id", values="clicked")
        .fillna(False)
        .to_numpy()
    )

    return R, X
