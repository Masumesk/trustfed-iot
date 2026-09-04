import torch


def compute_model_update(
    global_model,
    local_model
):

    global_parameters = dict(
        global_model.named_parameters()
    )

    total_numel = sum(
        parameter.numel()
        for parameter
        in local_model.parameters()
    )

    first_parameter = next(
        local_model.parameters()
    )

    update_vector = torch.empty(
        total_numel,
        dtype=first_parameter.dtype,
        device="cpu"
    )

    offset = 0

    for (
        name,
        local_parameter
    ) in local_model.named_parameters():

        global_parameter = (
            global_parameters[name]
        )

        numel = (
            local_parameter.numel()
        )

        target = update_vector[
            offset:
            offset + numel
        ]

        local_cpu = (
            local_parameter
            .detach()
            .cpu()
            .reshape(-1)
        )

        global_cpu = (
            global_parameter
            .detach()
            .cpu()
            .reshape(-1)
        )

        torch.sub(
            local_cpu,
            global_cpu,
            out=target
        )

        offset += numel

    return update_vector.numpy()

def apply_model_update(
    model,
    update_vector
):

    parameters = list(
        model.parameters()
    )

    if not parameters:
        return model

    device = parameters[0].device
    dtype = parameters[0].dtype

    flat_update = torch.as_tensor(
        update_vector,
        dtype=dtype,
        device=device
    )

    offset = 0

    with torch.no_grad():

        for parameter in parameters:

            numel = parameter.numel()

            update_part = (
                flat_update[
                    offset:
                    offset + numel
                ]
                .view_as(parameter)
            )

            parameter.add_(
                update_part
            )

            offset += numel

    if offset != len(update_vector):
        raise ValueError(
            "Update vector size "
            "does not match model."
        )

    return model