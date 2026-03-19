import ast
import logging
import os
from types import SimpleNamespace

import torch

from flpoison.aggregators import all_aggregators
from flpoison.attackers import data_poisoning_attacks, model_poisoning_attacks
from flpoison.fl.eval_schedule import DEFAULT_EVAL_INTERVAL
from flpoison.utils.config_utils import (
    load_dataset_catalog,
    load_experiment_config,
    resolve_config_path,
    validate_experiment_config,
)
from flpoison.utils.global_utils import (
    flush_bootstrap_logs,
    frac_or_int_to_int,
    queue_bootstrap_log,
    setup_console_logger,
)


KNOWN_ATTACKS = ['NoAttack'] + model_poisoning_attacks + data_poisoning_attacks
BOOTSTRAP_LOGGER_NAME = "flpoison.bootstrap"


def read_yaml(filename):
    resolved = resolve_config_path(filename)
    args_dict = load_experiment_config(resolved)
    validate_experiment_config(
        args_dict,
        source=resolved,
        known_attacks=KNOWN_ATTACKS,
        known_defenses=all_aggregators,
    )
    args = SimpleNamespace(**args_dict)
    return args


def _lookup_component_params(args, param_type, selected_name):
    for item in getattr(args, f"{param_type}s", []):
        if item.get(param_type) == selected_name:
            return item.get(f"{param_type}_params")
    return None


def override_args(args, cli_args):
    """
    1. fill the attack and defense parameters with default if not provided.
    2. override the arguments with provided command line arguments if possible.
    if attack and defense are provided:
        if their corresponding parameters provided:
            override them with the provided parameters
        else:
            override them with default attack parameters
    Args:
        args: the configuration object readin from the yaml file
        cli_args: the command line arguments
    """
    # fill the attack and defense parameters with default
    for param_type in ['attack', 'defense']:
        if not hasattr(args, f"{param_type}_params"):
            setattr(
                args,
                f"{param_type}_params",
                _lookup_component_params(args, param_type, getattr(args, param_type)),
            )

    # override parameters
    # if only attack or defense is provided, set their corresponding params to default
    for key, value in vars(cli_args).items():
        if key in ['config', 'attack', 'defense', 'attack_params', 'defense_params']:
            continue
        if value is not None:
            setattr(args, key, value)
            # keep stdout clean for experiment-control knobs
            if key not in ['num_experiments', 'experiment_id']:
                queue_bootstrap_log(args, logging.WARNING, f"Overriding {key} with {value}")

    # override attack, defense, attack_params, defense_params
    for param_type in ['attack', 'defense']:
        selected_name = getattr(cli_args, param_type)
        if selected_name:  # if not None
            setattr(args, param_type, selected_name)
            # if attack_params or defense_params is provided by cli_args, override the corresponding params
            selected_params = getattr(cli_args, f"{param_type}_params")
            if selected_params:
                setattr(args, f'{param_type}_params',
                        ast.literal_eval(selected_params))
            else:
                setattr(
                    args,
                    f"{param_type}_params",
                    _lookup_component_params(args, param_type, selected_name),
                )


def benchmark_preprocess(args):
    bootstrap_logger = setup_console_logger(
        BOOTSTRAP_LOGGER_NAME,
        color=getattr(args, "log_color", "auto"),
    )
    for attack_i in args.attacks:
        for defense_j in args.defenses:
            args.attack, args.attack_params = attack_i['attack'], attack_i.get(
                'attack_params')
            args.defense, args.defense_params = defense_j['defense'], defense_j.get(
                'defense_params')
            single_preprocess(args)
            flush_bootstrap_logs(args, bootstrap_logger)
            if os.path.exists(args.output):
                bootstrap_logger.info(
                    "File %s exists, skip",
                    args.output.split('/')[-1],
                )
                continue
            bootstrap_logger.info(
                "Running %s with %s under %s",
                args.attack,
                args.defense,
                args.distribution,
            )


def single_preprocess(args):
    dataset_config = load_dataset_catalog()
    for key, value in dataset_config[args.dataset].items():
        if key in ['mean', 'std']:
            value = tuple(value)
        setattr(args, key, value)

    # preprocess the arguments
    # Priority: CUDA > MPS (MacOS) > CPU
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_idx[0]}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    args.device = device
    # ensure optional flags exist
    if not hasattr(args, 'log_stream') or args.log_stream is None:
        args.log_stream = True
    if not hasattr(args, 'log_color') or args.log_color is None:
        args.log_color = "auto"
    args.num_adv = frac_or_int_to_int(args.num_adv, args.num_clients)
    if not hasattr(args, 'eval_batch_size') or args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    if not hasattr(args, 'eval_interval') or args.eval_interval is None:
        args.eval_interval = DEFAULT_EVAL_INTERVAL
    if not hasattr(args, 'record_time') or args.record_time is None:
        args.record_time = False
    if not hasattr(args, 'cudnn_benchmark') or args.cudnn_benchmark is None:
        args.cudnn_benchmark = args.device.type == "cuda"
    if not hasattr(args, 'allow_tf32') or args.allow_tf32 is None:
        args.allow_tf32 = False
    if not hasattr(args, 'torch_profile') or args.torch_profile is None:
        args.torch_profile = False
    if not hasattr(args, 'gpu_sample_interval_ms') or args.gpu_sample_interval_ms is None:
        args.gpu_sample_interval_ms = 100
    if not hasattr(args, 'torch_profile_wait') or args.torch_profile_wait is None:
        args.torch_profile_wait = 0
    if not hasattr(args, 'torch_profile_warmup') or args.torch_profile_warmup is None:
        args.torch_profile_warmup = 1
    if not hasattr(args, 'torch_profile_active') or args.torch_profile_active is None:
        args.torch_profile_active = 3
    if not hasattr(args, 'torch_profile_repeat') or args.torch_profile_repeat is None:
        args.torch_profile_repeat = 1
    if not hasattr(args, 'torch_profile_record_shapes') or args.torch_profile_record_shapes is None:
        args.torch_profile_record_shapes = True
    if not hasattr(args, 'torch_profile_memory') or args.torch_profile_memory is None:
        args.torch_profile_memory = True
    if not hasattr(args, 'torch_profile_with_stack') or args.torch_profile_with_stack is None:
        args.torch_profile_with_stack = False

    # ensure attack_params and defense_params attributes exist. when there is no params, set it to None.
    ensure_attr(args, 'attack_params')
    ensure_attr(args, 'defense_params')

    # generate output path if not provided
    if not hasattr(args, 'output') or args.output is None:
        args.output = f'./logs/{args.algorithm}/{args.dataset}_{args.model}/{args.distribution}/{args.dataset}_{args.model}_{args.distribution}_{args.attack}_{args.defense}_{args.epochs}_{args.num_clients}_{args.learning_rate}_{args.algorithm}.txt'

    # check output path, if exists, skip, otherwise create the directories
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    return args


def ensure_attr(obj, attr_name):
    """
    set attr_name of obj to None if it does not exist
    """
    if not hasattr(obj, attr_name):
        setattr(obj, attr_name, None)
