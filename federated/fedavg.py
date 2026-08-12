import copy


def fedavg(local_models, sample_counts):

    total_samples = sum(sample_counts)

    global_state = copy.deepcopy(
        local_models[0].state_dict()
    )

    for key in global_state.keys():

        global_state[key] = (
            global_state[key] * 0
        )

        for model, num_samples in zip(
            local_models,
            sample_counts
        ):
            weight = num_samples / total_samples

            global_state[key] += (
                weight * model.state_dict()[key]
            )

    return global_state