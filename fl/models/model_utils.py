from copy import deepcopy
import numpy as np
import torch

# ======The two main APIs for model and 1-d numpy array conversion======

def _ensure_tensor(vector, device=None, dtype=None):
    if torch.is_tensor(vector):
        tensor = vector
    else:
        tensor = torch.from_numpy(vector)
    if device is not None or dtype is not None:
        tensor = tensor.to(
            device=device if device is not None else tensor.device,
            dtype=dtype if dtype is not None else tensor.dtype,
        )
    return tensor


def vec2model(vector, model, plus=False, ignorebn=False):
    """
    in-place modification of model's parameters/buffers
    Convert a 1d vector back into a model
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    vec_tensor = _ensure_tensor(vector, device=device, dtype=dtype)

    # ---- build or reuse cached layout ----
    # layout element: (key, numel, shape)
    # cache is keyed by ignorebn (because skip set differs)
    cache_attr = "_vec2model_layout_ignorebn" if ignorebn else "_vec2model_layout"

    layout = getattr(model, cache_attr, None)
    if layout is None:
        state = model.state_dict()
        bn_skip = ('running_mean', 'running_var', 'num_batches_tracked')

        built = []
        for k, v in state.items():
            if ignorebn and any(s in k for s in bn_skip):
                continue
            built.append((k, v.numel(), v.shape))
        layout = built
        setattr(model, cache_attr, layout)

    # ---- apply vector into model ----
    curr_idx = 0
    state = model.state_dict()  # lightweight handle; tensors are references

    with torch.no_grad():
        for k, numel, shape in layout:
            value = state[k]
            param_tensor = vec_tensor[curr_idx:curr_idx + numel].view(shape)

            if plus:
                value.add_(param_tensor)
            else:
                value.copy_(param_tensor)

            curr_idx += numel


def model2vec(model):
    """
    convert the model's state dict to a 1d numpy array
    """
    return state2vec(model.state_dict())
    # return parameter2vector(model)


def add_vec2model(vector, model_template):
    """
    Add the state dict vector-form (pseudo) gradient to the model's parameters and return a new model
    """
    tmp_model = deepcopy(model_template)
    vec2model(vector, tmp_model, plus=True)
    return tmp_model

# ======Below are specific implementations======


def vector2parameter(vector, model):
    """
    in-place modification of iterable model_parameters's data
    """
    current_pos = 0
    params = list(model.parameters())
    if not params:
        return
    vec_tensor = _ensure_tensor(
        vector, device=params[0].device, dtype=params[0].dtype)
    for param in params:
        numel = param.numel()  # get the number of elements in param
        param.data.copy_(
            vec_tensor[current_pos:current_pos + numel].reshape(param.shape))
        current_pos += numel


def parameter2vector(model):
    # numpy() will convert torch.float32 to np.float32
    params = list(model.parameters())
    if not params:
        return np.array([])
    vec = torch.cat([param.detach().reshape(-1) for param in params])
    return vec.cpu().numpy()


def set_grad_none(model):
    """
    Set the gradient of all parameters to None
    """
    for param in model.parameters():
        param.grad = None


def vector2gradient(vector, model):
    """
    in-place modification of iterable model_parameters's grad
    """
    current_pos = 0
    params = list(model.parameters())
    if not params:
        return
    vec_tensor = _ensure_tensor(
        vector, device=params[0].device, dtype=params[0].dtype)
    for param in params:
        numel = param.numel()  # get the number of elements in param
        param.grad = vec_tensor[current_pos:current_pos +
                                numel].reshape(param.shape)
        current_pos += numel


def gradient2vector(model):
    """
    Convert gradients to a concatenated 1D numpy array
    """
    params = list(model.parameters())
    if not params:
        return np.array([])
    vec = torch.cat([param.grad.detach().reshape(-1) for param in params])
    return vec.cpu().numpy()


def ol_from_vector(vector, model_template, flatten=True, return_type='dict'):
    state_template = model_template.state_dict()
    # Get keys for the last two layers (weight and bias)
    output_layer_keys = list(state_template.keys())[-2:]

    # Get the shapes of the weight and bias
    weight_shape = state_template[output_layer_keys[0]].shape
    bias_shape = state_template[output_layer_keys[1]].shape

    # Calculate sizes
    weight_size = np.prod(weight_shape)
    bias_size = np.prod(bias_shape)

    # Start with the last element of the vector for bias, then weight
    bias = vector[-bias_size:
                  ] if flatten else vector[-bias_size:].reshape(bias_shape)
    weights = vector[-(bias_size + weight_size):-bias_size] if flatten else vector[-(bias_size + weight_size):-
                                                                                   bias_size].reshape(weight_shape)
    if return_type == 'dict':
        return {'weight': weights, 'bias': bias}
    elif return_type == 'vector':
        # !DON'T change the order of weights and bias, as it's the order of the output layer and the order of the state_dict vector
        if flatten:  # concatenate 1d vectors
            return np.concatenate([weights.flatten(), bias.flatten()])
        else:
            # concatenate the weights and bias at axis 1, i.e., column-wise, to produce a 2d array with same number of rows and added bias columns
            return np.concatenate([weights, bias.reshape(bias_size, -1)], axis=1)


def ol_from_model(model, flatten=True, return_type='dict'):
    return ol_from_vector(model2vec(model), model,
                          flatten=flatten, return_type=return_type)


def vec2state(vector, model, plus=False, ignorebn=False, numpy=False):
    """
    Convert a 1d-numpy array to the state dict-form of the model
    return a new state dict
    """
    curr_idx = 0
    state = model.state_dict()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    vec_tensor = _ensure_tensor(vector, device=device, dtype=dtype)

    new_state = {}

    with torch.no_grad():
        for key, value in state.items():
            if ignorebn and any(substring in key for substring in ['running_mean', 'running_var', 'num_batches_tracked']):
                new_state[key] = value
                continue

            numel = value.numel()
            param_tensor = vec_tensor[curr_idx:curr_idx + numel].view_as(value)

            if plus:
                new_state[key] = value + param_tensor
            else:
                new_state[key] = param_tensor.clone()

            curr_idx += numel
    
    if numpy:
        return {key: value.detach().cpu().numpy() for key, value in new_state.items()}

    return new_state


def state2vec(model_state_dict, ignorebn=False, numpy_flg=False, return_torch=False):
    """
    Convert a state dict to a concatenated 1D numpy array.
    """
    # Collect all the tensors first
    if numpy_flg:
        arrays = [
            value.flatten()
            for name, value in model_state_dict.items()
            if (True if not ignorebn else all(substring not in name for substring in ['running_mean', 'running_var', 'num_batches_tracked']))
        ]
        return np.concatenate(arrays) if arrays else np.array([])

    tensors = [
        i.detach().reshape(-1)
        for name, i in model_state_dict.items()
        if (True if not ignorebn else all(substring not in name for substring in ['running_mean', 'running_var', 'num_batches_tracked']))
    ]
    if not tensors:
        return torch.empty(0) if return_torch else np.array([])
    vec = torch.cat(tensors)
    if return_torch:
        return vec
    # Concatenate the list of arrays at once
    return vec.cpu().numpy()
