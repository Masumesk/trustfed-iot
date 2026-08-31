import numpy as np


def _get_client_info(client_id, client_infos):

    if client_id in client_infos:
        return client_infos[client_id]

    return client_infos[str(client_id)]


def _aggregate_distribution(client_ids, client_infos):


    first_client = next(iter(client_infos.values()))

    num_classes = len(
        first_client["distribution"]
    )

    class_counts = np.zeros(
        num_classes,
        dtype=float
    )

    for client_id in client_ids:

        info = _get_client_info(
            client_id,
            client_infos
        )

        distribution = np.asarray(
            info["distribution"],
            dtype=float
        )

        num_samples = float(
            info["num_samples"]
        )

        # Approximate number of samples
        # belonging to each class
        class_counts += (
            distribution * num_samples
        )

    total = class_counts.sum()

    if total == 0:
        return np.zeros(
            num_classes,
            dtype=float
        )

    return class_counts / total


def calculate_representation_fairness(
    client_ids,
    client_infos
):

    if not client_ids:
        return {
            "fairness": 0.0,
            "hellinger_distance": 1.0,
            "global_distribution": [],
            "selected_distribution": [],
        }

    # Distribution of all clients
    all_client_ids = list(
        client_infos.keys()
    )

    global_distribution = (
        _aggregate_distribution(
            all_client_ids,
            client_infos
        )
    )

    # Distribution of selected main clients
    selected_distribution = (
        _aggregate_distribution(
            client_ids,
            client_infos
        )
    )

    # Hellinger distance
    hellinger_distance = np.sqrt(
        0.5 * np.sum(
            (
                np.sqrt(global_distribution)
                -
                np.sqrt(selected_distribution)
            ) ** 2
        )
    )

    representation_fairness = (
        1.0 - hellinger_distance
    )

    return {
        "fairness":
            float(representation_fairness),

        "hellinger_distance":
            float(hellinger_distance),

        "global_distribution":
            global_distribution.tolist(),

        "selected_distribution":
            selected_distribution.tolist(),
    }