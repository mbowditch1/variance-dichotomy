import numpy as np
import pandas as pd
from scipy.stats import special_ortho_group
from tqdm import tqdm


def random_vector_pair(d, phi):
    R = special_ortho_group.rvs(d)

    x = np.dot(R, np.append(1, np.zeros(d - 1)))
    lower_sphere_sample = np.random.randn(d - 1)
    lower_sphere_sample /= np.linalg.norm(lower_sphere_sample)
    lower_sphere_sample *= np.sin(phi)
    y = np.append(np.cos(phi), lower_sphere_sample)

    y = np.dot(R, y)
    y /= np.linalg.norm(y)
    return x, y


def generate_vector_pair(d, matching, network, split=0, factor=1000):
    if matching:
        data_distribution = {
            "facenet": (0.2841283082962036, 0.13104695081710815),
            "facenet512": (0.25047004222869873, 0.12035150825977325),
            "adaface": (0.49594250321388245, 0.15808598697185516),
            "arcface": (0.4216248095035553, 0.14368464052677155),
        }
    else:
        data_distribution = {
            "facenet": (0.9326430559158325, 0.14588691294193268),
            "facenet512": (0.9722923040390015, 0.1552008092403412),
            "adaface": (0.9846383333206177, 0.05719079077243805),
            "arcface": (0.8578531742095947, 0.10590147227048874),
        }

    while True:
        phi = 1 - np.random.normal(*data_distribution[network])
        if phi <= 1 and phi >= -1:
            break
    phi = np.arccos(phi)

    if split <= 0:
        split = d

    x, y = random_vector_pair(split, phi)
    if split < d:
        xl, yl = random_vector_pair(d - split, phi)
        x = np.append(x, xl / factor)
        y = np.append(y, yl / factor)
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)

    return x, y


def compute_angles(x, y, split=0, factor=1000):
    theta = [np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y)]

    if split <= 0:
        split = d

    p = np.random.randn(split, d - 1)
    p /= np.linalg.norm(p, axis=0)
    if split < d:
        p = np.append(p, np.random.randn(d - split, d - 1) / factor, axis=0)
        p /= np.linalg.norm(p, axis=0)
    Q, _ = np.linalg.qr(p)
    # the columns of Q are now orthogonal and each column consitute one direction for our projection

    for i in range(d - 1):
        # project x and y onto the subspace orthogonal to Q[:,i]
        x = x - np.dot(x, Q[:, i]) * Q[:, i]
        y = y - np.dot(y, Q[:, i]) * Q[:, i]
        # renormalize
        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)

        # record the cosine of the angle between the two vectors
        theta.append(np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y))
    return theta


def statistic(d, eps, trials=100, network="facenet", split=0, factor=1000):
    count = [0] * d
    count2 = [0] * d

    for _ in tqdm(range(trials)):
        x, y = generate_vector_pair(
            d, True, network=network, split=split, factor=factor
        )
        theta = compute_angles(x, y, split=split, factor=factor)
        count = [i + 1 if t > np.cos(eps) else i for t, i in zip(theta, count)]

        x, y = generate_vector_pair(
            d, False, network=network, split=split, factor=factor
        )
        theta = compute_angles(x, y, split=split, factor=factor)
        count2 = [i + 1 if t < np.cos(eps) else i for t, i in zip(theta, count2)]

    df = pd.DataFrame(
        columns=["d", "eps", "trials", "l", "count", "count2", "benign accuracy"]
    )
    df["d"] = [d] * d
    df["eps"] = [eps] * d
    df["trials"] = [trials] * d
    df["l"] = range(len(count))
    df["count"] = count
    df["count2"] = count2
    df["benign accuracy"] = [
        (acc1 + acc2) / (2 * trials) for acc1, acc2 in zip(count, count2)
    ]

    return df


if __name__ == "__main__":
    factors = {"facenet": 1, "facenet512": 1, "arcface": 1, "adaface": 1}
    splits = {"facenet": 66, "facenet512": 49, "adaface": 404, "arcface": 174}
    eps = {"facenet": 1-0.3966, "facenet512": 1-0.4126, "adaface": 1-0.1556, "arcface": 1-0.3398}
    eps = {k: np.arccos(1 - v) for k, v in eps.items()}

    # for model_name, d in zip(["facenet", "facenet512", "arcface", "adaface"], [128, 512, 512, 512]):
    for model_name, d in zip(["adaface"], [512]):
        data = []
        print("d", d)
        data.append(
            statistic(
                d,
                eps[model_name],
                trials=1000,
                network=model_name,
                split=splits[model_name],
                factor=factors[model_name],
            )
        )

        pd.concat(data).to_csv(f"csv/{model_name}_simulation_1k.csv", index=False)
