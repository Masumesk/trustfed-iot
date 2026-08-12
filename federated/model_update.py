import torch


def compute_model_update(global_model, local_model):
    update_parts = []

    global_parameters = dict(
        global_model.named_parameters()
    )

    for name, local_parameter in local_model.named_parameters():

        global_parameter = global_parameters[name]

        difference = (
            local_parameter.detach().cpu()
            - global_parameter.detach().cpu()
        )

        update_parts.append(
            difference.reshape(-1)
        )

    update_vector = torch.cat(
        update_parts
    )

    return update_vector.numpy()