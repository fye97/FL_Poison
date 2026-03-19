import time
from functools import wraps
import importlib
import os
import logging
import random
import sys
import numpy as np
import torch
from tqdm import tqdm


class TqdmLoggingHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream or sys.stdout)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


class ColorFormatter(logging.Formatter):
    RESET = "\033[0m"
    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }

    def __init__(self, fmt='%(message)s', use_color=False):
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record):
        msg = super().format(record)
        if not self.use_color:
            return msg
        color = self.COLORS.get(record.levelno)
        if not color or not msg:
            return msg
        return f"{color}{msg}{self.RESET}"


class MaxLevelFilter(logging.Filter):
    def __init__(self, exclusive_upper_bound):
        super().__init__()
        self.exclusive_upper_bound = exclusive_upper_bound

    def filter(self, record):
        return record.levelno < self.exclusive_upper_bound


def normalize_log_color(value):
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "auto"}:
        return "auto"
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid log color mode: {value}")


def stream_supports_color(stream):
    if stream is None:
        return False
    try:
        is_tty = stream.isatty()
    except Exception:
        is_tty = False
    if not is_tty:
        return False
    return os.environ.get("TERM", "").lower() != "dumb"


def should_colorize_stream(color, stream):
    mode = normalize_log_color(color)
    if mode is False:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if mode is True or os.environ.get("FORCE_COLOR"):
        return True
    return stream_supports_color(stream)


def clear_logger_handlers(logger):
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass


def build_stream_handler(*, use_tqdm=False, stream=None, formatter=None):
    if use_tqdm:
        handler = TqdmLoggingHandler(stream=stream or sys.stdout)
    else:
        handler = logging.StreamHandler(stream)
    if formatter is not None:
        handler.setFormatter(formatter)
    return handler


def setup_logger(
    logger_name,
    log_file=None,
    level=logging.INFO,
    stream=False,
    use_tqdm=False,
    color="auto",
    stream_target=None,
):
    logger = logging.getLogger(logger_name)
    # In repeated experiments within the same Python process, avoid accumulating handlers.
    clear_logger_handlers(logger)
    logger.propagate = False
    plain_formatter = logging.Formatter('%(message)s')

    logger.setLevel(level)

    if log_file is not None:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fileHandler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fileHandler.setFormatter(plain_formatter)
        logger.addHandler(fileHandler)

    if stream:
        default_stream = stream_target
        if default_stream is None and use_tqdm:
            default_stream = sys.stdout
        stream_formatter = ColorFormatter(
            '%(message)s',
            use_color=should_colorize_stream(color, default_stream or sys.stderr),
        )
        streamHandler = build_stream_handler(
            use_tqdm=use_tqdm,
            stream=default_stream,
            formatter=stream_formatter,
        )
        logger.addHandler(streamHandler)

    return logger


def setup_console_logger(logger_name, level=logging.INFO, color="auto"):
    logger = logging.getLogger(logger_name)
    clear_logger_handlers(logger)
    logger.propagate = False
    logger.setLevel(level)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(MaxLevelFilter(logging.WARNING))
    stdout_handler.setFormatter(
        ColorFormatter('%(message)s', use_color=should_colorize_stream(color, sys.stdout))
    )
    logger.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(
        ColorFormatter('%(message)s', use_color=should_colorize_stream(color, sys.stderr))
    )
    logger.addHandler(stderr_handler)
    return logger


def get_context_logger(source=None, logger_name=None, level=logging.INFO, color="auto"):
    if isinstance(source, logging.Logger):
        return source

    if source is not None:
        source_logger = getattr(source, "logger", None)
        if isinstance(source_logger, logging.Logger):
            return source_logger
        color = getattr(source, "log_color", color)

    resolved_name = logger_name or __name__
    logger = logging.getLogger(resolved_name)
    if logger.handlers:
        return logger
    return setup_console_logger(resolved_name, level=level, color=color)


def queue_bootstrap_log(args, level, message):
    bootstrap_logs = getattr(args, "_bootstrap_logs", None)
    if bootstrap_logs is None:
        bootstrap_logs = []
        setattr(args, "_bootstrap_logs", bootstrap_logs)
    bootstrap_logs.append((int(level), str(message)))


def flush_bootstrap_logs(args, logger):
    for level, message in getattr(args, "_bootstrap_logs", []):
        logger.log(level, message)
    if hasattr(args, "_bootstrap_logs"):
        args._bootstrap_logs = []


def actor(category, *attributes):
    """class decorator for categorizing attackers, data poisoners (backdoor, others), and model poisoners (omniscient, non_omniscient, others)
    """
    def decorator(cls):
        # key is the actor, value is the attributes of the actor
        categories = {"benign": ['always', 'temporary'],
                      "attacker": ['data_poisoning', 'model_poisoning', "non_omniscient", "omniscient"]}

        if not set(attributes).issubset(set(categories[category])):
            raise ValueError(
                "Invalid sub-category. Please change or add the sub-category.")
        cls._category = category
        cls._attributes = attributes
        # change __init__ method to realize it in objects
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            # Check if the object has already been decorated to avoid redundant decoration due to inheritance.
            # When attacker init, one class, two different object (attacker, super init)
            if not hasattr(self, "_decorated"):
                self.category = cls._category
                self.attributes = cls._attributes
                # Mark self object as decorated to prevent re-decoration on inherited classes
                self._decorated = True
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls
    return decorator


def import_all_modules(current_dir, depth=0, depth_prefix=None, package_name=None):
    if package_name is not None:
        pkg_name = package_name
    else:
        pkg_name = depth_prefix + "." + os.path.basename(
            current_dir) if depth else os.path.basename(current_dir)

    for filename in os.listdir(current_dir):
        # filter our __init__.py and non-python files
        if filename.endswith(".py") and (filename != "__init__.py"):
            module_name = filename[:-3]  # remove ".py"
            importlib.import_module(
                f".{module_name}", package=pkg_name)


class Register(dict):
    """Register class is a dict class with 2 functions: 1. serve as the registry 2. register the callable object, function, to this registry
    """

    def __init__(self, *args, **kwargs):
        # init the dict class, so that it can be used as a normal dict
        super().__init__(*args, **kwargs)

    def __call__(self, target):
        def register_item(name, func):
            self[name] = func
            return func

        # if target is a string, return a function to receive the callable object. @register('name')
        if isinstance(target, str):
            ret_func = (lambda x: register_item(target, x))
        # if target is a callable object, then register it, and return it, @register
        elif callable(target):
            ret_func = register_item(target.__name__, target)
        return ret_func


def print_filtered_args(args, logger):
    args_dict = vars(args)
    hidden_keys = {'attacks', 'defenses', 'logger', '_bootstrap_logs'}
    remaining = {
        key: value for key, value in args_dict.items()
        if key not in hidden_keys
    }
    sections = [
        (
            "Run",
            [
                "seed",
                "seed_start",
                "num_experiments",
                "experiment_id",
                "epochs",
                "algorithm",
                "optimizer",
                "momentum",
                "weight_decay",
                "learning_rate",
                "lr_scheduler",
                "milestones",
                "local_epochs",
                "eval_interval",
            ],
        ),
        (
            "Data",
            [
                "dataset",
                "model",
                "distribution",
                "dirichlet_alpha",
                "im_iid_gamma",
                "tail_cls_from",
                "num_training_sample",
                "num_channels",
                "num_classes",
                "batch_size",
                "eval_batch_size",
                "num_clients",
                "num_adv",
                "cache_partition",
                "mean",
                "std",
            ],
        ),
        (
            "Attack / Defense",
            [
                "attack",
                "attack_params",
                "defense",
                "defense_params",
            ],
        ),
        (
            "Runtime",
            [
                "device",
                "gpu_idx",
                "num_workers",
                "log_stream",
                "log_color",
                "record_time",
                "cudnn_benchmark",
                "allow_tf32",
                "torch_profile",
                "gpu_sample_interval_ms",
                "torch_profile_wait",
                "torch_profile_warmup",
                "torch_profile_active",
                "torch_profile_repeat",
                "torch_profile_record_shapes",
                "torch_profile_memory",
                "torch_profile_with_stack",
            ],
        ),
        (
            "Output",
            [
                "output",
                "root",
                "aug",
                "partition_visualization",
            ],
        ),
    ]

    lines = ["Configuration Summary"]
    for title, keys in sections:
        items = []
        for key in keys:
            if key in remaining:
                items.append((key, remaining.pop(key)))
        if not items:
            continue
        width = max(len(key) for key, _ in items)
        lines.append(f"[{title}]")
        for key, value in items:
            lines.append(f"  {key:<{width}} : {value}")
        lines.append("")

    if remaining:
        other_items = sorted(remaining.items())
        width = max(len(key) for key, _ in other_items)
        lines.append("[Other]")
        for key, value in other_items:
            lines.append(f"  {key:<{width}} : {value}")

    logger.info("\n%s", "\n".join(lines).rstrip())


def avg_value(x):
    return sum(x) / len(x)


def setup_seed(seed):
    """
    fix all possible randomness for reproduction
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def frac_or_int_to_int(frac_or_int, total_num):
    return int(frac_or_int) if frac_or_int >= 1 else int(frac_or_int * total_num)


class TimingRecorder:
    def __init__(self, id, output_file):
        self.id = id
        # record the duration and number of call of func
        self.global_timings = {}
        time_log_path = output_file.replace(
            "logs/", "logs/time_logs/", 1)[:-4]+'.log'
        self.logger = setup_logger(
            __name__, time_log_path, level=logging.INFO)
        self.client_log_flag = False
        epoch_level = False
        self.record_epochs = [2, 4, 6, 8, 10, 20,
                              50, 100, 150, 200] if epoch_level else []

    def timing_decorator(self, func):
        """decorator to record the running time of each function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()  # start timer
            result = func(*args, **kwargs)  # call the function
            end_time = time.time()  # end timer
            duration = end_time - start_time

            # update the global_timings with the duration
            method_name = func.__name__
            if method_name not in self.global_timings:
                self.global_timings[method_name] = {
                    "total_time": 0, "calls": 0}
            self.global_timings[method_name]["total_time"] += duration
            self.global_timings[method_name]["calls"] += 1

            if self.client_log_flag:
                # log data during training
                self.report(f"Worker ID {self.id}")

            # for client
            self.client_log_flag = True if method_name == "local_training" and self.global_timings[
                method_name]["calls"] in self.record_epochs else False

            if method_name == "aggregation" and self.global_timings[method_name]["calls"] in self.record_epochs:
                # log data during training
                self.report(f"Worker ID {self.id}")
            return result
        return wrapper

    def get_average_time(self, func_name):
        """get average running time of func_name for all epoch"""
        if func_name in self.global_timings:
            total_time = self.global_timings[func_name]["total_time"]
            calls = self.global_timings[func_name]["calls"]
            return total_time / calls if calls > 0 else 0
        return 0

    def report(self, id=None):
        for method_name, stats in self.global_timings.items():
            avg_time = stats["total_time"] / stats["calls"]
            self.logger.info(
                f"{id}, {method_name} averge time: {avg_time:.6f} s, call time: {stats['calls']}")
