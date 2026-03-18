import argparse
import sys
from types import SimpleNamespace

from flpoison.aggregators import all_aggregators
from flpoison.fl.algorithms import all_algorithms
from flpoison.fl.configuration import KNOWN_ATTACKS, read_yaml
from flpoison.fl.models import all_models


def _normalize_single_dash_long_opts(argv, parser):
    """
    Support passing long options with a single dash, e.g.:
      -experiment_id=0  -> --experiment_id=0
      -num_experiments 5 -> --num_experiments 5

    This is needed because argparse treats "-experiment_id" as "-e xperiment_id"
    when "-e/--epochs" exists, so exact single-dash long options are unreliable
    without normalization.
    """
    long_names = set()
    for opt in parser._option_string_actions.keys():
        if opt.startswith("--") and len(opt) > 2:
            long_names.add(opt[2:])

    out = []
    for tok in argv:
        if tok.startswith("--") or not tok.startswith("-") or tok == "-":
            out.append(tok)
            continue

        if len(tok) <= 2:
            out.append(tok)
            continue

        body = tok[1:]
        name = body.split("=", 1)[0]
        if name in long_names:
            out.append("--" + body)
        else:
            out.append(tok)
    return out


def read_args():
    """
    1. Parse command-line arguments for configuration path and possible overrides.
    2. Load the experiment preset from the provided YAML file.
    3. Return the base configuration plus CLI overrides.
    """
    parser = argparse.ArgumentParser(
        description="Poisoning attacks and defenses in Federated Learning"
    )
    parser.add_argument(
        "-config",
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "-b",
        "-benchmark",
        "--benchmark",
        action="store_true",
        default=None,
        help="Run all combinations of attacks and defenses",
    )
    parser.add_argument("-e", "-epochs", "--epochs", type=int)
    parser.add_argument("-seed", "--seed", type=int)
    parser.add_argument(
        "-num_experiments",
        "--num_experiments",
        type=int,
        default=None,
        help="Number of repeated experiments (seeds increment by 1 each run)",
    )
    parser.add_argument(
        "-experiment_id",
        "--experiment_id",
        type=int,
        default=None,
        help="Starting experiment id (default: 0). Effective seed = seed_start + experiment_id",
    )
    parser.add_argument("-alg", "-algorithm", "--algorithm", choices=all_algorithms)
    parser.add_argument(
        "-opt",
        "-optimizer",
        "--optimizer",
        choices=["SGD", "Adam"],
        help="optimizer for training",
    )
    parser.add_argument("-lr_scheduler", "--lr_scheduler", type=str, help="lr_scheduler for training")
    parser.add_argument("-milestones", "--milestones", type=int, nargs="+", help="milestone for learning rate scheduler")
    parser.add_argument("-num_clients", "--num_clients", type=int, help="number of participating clients")
    parser.add_argument("-bs", "-batch_size", "--batch_size", type=int, help="batch_size")
    parser.add_argument(
        "-eval_batch_size",
        "--eval_batch_size",
        type=int,
        help="batch_size used for evaluation/inference",
    )
    parser.add_argument("-lr", "-learning_rate", "--learning_rate", type=float, help="initial learning rate")
    parser.add_argument("-le", "-local_epochs", "--local_epochs", type=int, help="local global_epoch")
    parser.add_argument("-eval_interval", "--eval_interval", type=int, help="Run full evaluation every N global rounds")
    parser.add_argument("-model", "--model", choices=all_models)
    parser.add_argument(
        "-data",
        "-dataset",
        "--dataset",
        choices=["MNIST", "FashionMNIST", "CIFAR10", "CINIC10", "CIFAR100", "EMNIST", "CHMNIST", "TinyImageNet", "HAR"],
    )
    parser.add_argument(
        "-dtb",
        "-distribution",
        "--distribution",
        choices=["iid", "class-imbalanced_iid", "non-iid", "pat", "imbalanced_pat"],
    )
    parser.add_argument(
        "-dirichlet_alpha",
        "--dirichlet_alpha",
        type=float,
        help="smaller alpha for drichlet distribution, stronger heterogeneity, 0.1 0.5 1 5 10, normally use 0.5",
    )
    parser.add_argument(
        "-im_iid_gamma",
        "--im_iid_gamma",
        type=float,
        help="smaller alpha for class imbalanced distribution, stronger heterogeneity, 0.05, 0.1, 0.5",
    )

    parser.add_argument("-att", "-attack", "--attack", choices=KNOWN_ATTACKS, help="Attacks options")
    parser.add_argument("-attack_start_epoch", "--attack_start_epoch", type=int, help="the attack start epoch")
    parser.add_argument(
        "-attparam",
        "--attparam",
        type=float,
        help="scale for omniscient model poisoning attack, IPM,ALIE,MinMax,MinSum,Fang",
    )
    parser.add_argument("-def", "-defense", "--defense", choices=all_aggregators, help="Defenses options")
    parser.add_argument(
        "-num_adv",
        "--num_adv",
        type=float,
        help="the proportion (float < 1) or number (int>1) of adversaries",
    )
    parser.add_argument("-o", "-output", "--output", type=str, help="output file for results")
    parser.add_argument(
        "-log_stream",
        "--log_stream",
        action="store_true",
        default=None,
        help="Enable logging to stdout (tqdm-safe).",
    )
    parser.add_argument(
        "--log_color",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Colorize console logs. Default: auto-detect only on TTY streams.",
    )
    parser.add_argument(
        "-record_time",
        "--record_time",
        action="store_true",
        default=None,
        help="Enable runtime performance timing and per-round summaries.",
    )
    parser.add_argument(
        "-torch_profile",
        "--torch_profile",
        action="store_true",
        default=None,
        help="Enable torch.profiler tracing.",
    )
    parser.add_argument(
        "-gpu_sample_interval_ms",
        "--gpu_sample_interval_ms",
        type=int,
        help="GPU sampling interval in milliseconds when runtime timing is enabled.",
    )
    parser.add_argument("-torch_profile_wait", "--torch_profile_wait", type=int, help="torch.profiler schedule wait steps.")
    parser.add_argument("-torch_profile_warmup", "--torch_profile_warmup", type=int, help="torch.profiler schedule warmup steps.")
    parser.add_argument("-torch_profile_active", "--torch_profile_active", type=int, help="torch.profiler schedule active steps.")
    parser.add_argument("-torch_profile_repeat", "--torch_profile_repeat", type=int, help="torch.profiler schedule repeat count.")
    parser.add_argument(
        "-torch_profile_record_shapes",
        "--torch_profile_record_shapes",
        action="store_true",
        default=None,
        help="Record operator shapes in torch.profiler.",
    )
    parser.add_argument(
        "-torch_profile_memory",
        "--torch_profile_memory",
        action="store_true",
        default=None,
        help="Record memory usage in torch.profiler.",
    )
    parser.add_argument(
        "-torch_profile_with_stack",
        "--torch_profile_with_stack",
        action="store_true",
        default=None,
        help="Capture Python stacks in torch.profiler.",
    )
    parser.add_argument(
        "-prate",
        "-poisoning_ratio",
        "--poisoning_ratio",
        help="poisoning portion (float, range from 0 to 1, default: 0.1)",
    )
    parser.add_argument(
        "-target_label",
        "--target_label",
        type=int,
        help="The No. of target label for backdoored images (int, range from 0 to 10, default: 6)",
    )
    parser.add_argument("-trigger_path", "--trigger_path", help="Trigger Path")
    parser.add_argument("-trigger_size", "--trigger_size", type=int, help="Trigger Size (int, default: 5)")
    parser.add_argument(
        "-gidx",
        "-gpu_idx",
        "--gpu_idx",
        type=int,
        nargs="+",
        help="Index of GPU (int, default: 3, choice: 0, 1, 2, 3...)",
    )
    parser.add_argument("-num_workers", "--num_workers", type=int, help="Number of dataloader workers")
    parser.add_argument("-defense_params", "--defense_params", type=str, help="Override defense parameters")
    parser.add_argument("-attack_params", "--attack_params", type=str, help="Override attack parameters")

    cli_args = parser.parse_args(args=_normalize_single_dash_long_opts(sys.argv[1:], parser))
    args = SimpleNamespace()
    if cli_args.config:
        args = read_yaml(cli_args.config)
    return args, cli_args
