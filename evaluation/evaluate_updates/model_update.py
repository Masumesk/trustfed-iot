import torch
from torch.nn.utils import parameters_to_vector


def apply_model_update(model, update_vector):

    parameters = list(model.parameters())

    if not parameters:
        return model

    device = parameters[0].device
    dtype = parameters[0].dtype

    flat_update = torch.as_tensor(update_vector, dtype=dtype, device=device)
    offset = 0

    with torch.no_grad():

        for parameter in parameters:

            numel = parameter.numel()

            update_part = flat_update[offset : offset + numel].view_as(parameter)

            parameter.add_(update_part)

            offset += numel

    if offset != len(update_vector):
        raise ValueError("Update vector size does not match model.")

    return model


def state_dict_to_parameter_vector(global_state_dict, local_model):

    named_parameters = list(local_model.named_parameters())

    if not named_parameters:
        return torch.empty(0)

    return torch.cat(
        [global_state_dict[name].detach().reshape(-1) for name, _ in named_parameters]
    )


def compute_model_update_from_state_dict(
    global_state_dict, local_model, global_vector=None
):

    named_parameters = list(local_model.named_parameters())

    if not named_parameters:
        return torch.empty(0).numpy()

    local_vector = parameters_to_vector(
        [parameter.detach() for _, parameter in named_parameters]
    )

    if global_vector is None:
        global_vector = state_dict_to_parameter_vector(
            global_state_dict,
            local_model,
        )

    update_vector = local_vector - global_vector

    return update_vector.detach().cpu().contiguous().numpy()
