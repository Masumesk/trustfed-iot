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

def apply_model_update(
    model,
    update_vector
):
    offset = 0

    with torch.no_grad():

        for parameter in model.parameters():

            numel = parameter.numel()

            update_part = update_vector[
                offset:offset + numel
            ]

            update_tensor = torch.tensor(
                update_part,
                dtype=parameter.dtype,
                device=parameter.device
            ).reshape(parameter.shape)

            parameter.add_(update_tensor)

            offset += numel

    if offset != len(update_vector):
        raise ValueError(
            "Update vector size does not match model."
        )

    return model