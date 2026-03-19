import gc
import logging
import re
import time
from pathlib import Path

import torch
from tqdm import tqdm

from flpoison.fl import coordinator
from flpoison.datapreprocessor.data_utils import load_data, split_dataset
from flpoison.fl.configuration import benchmark_preprocess, override_args, single_preprocess
from flpoison.fl.eval_schedule import should_run_evaluation
from flpoison.fl.server import Server
from flpoison.utils.global_utils import (
    flush_bootstrap_logs,
    print_filtered_args,
    setup_logger,
    setup_seed,
)
from flpoison.utils.plot_utils import plot_accuracy
from flpoison.utils.performance_utils import RuntimeProfiler, create_torch_profiler, summarize_torch_profiler


def _output_with_experiment_id(base_output: str, seed: int, experiment_id: int) -> str:
    """
    Ensure each experiment writes to a unique output file.
    - If base_output already contains a seed token like "seed0" or "_seed0_", rewrite it to the effective seed.
    - If base_output already contains an experiment token like "exp0" or "_exp0_", rewrite it to the current experiment id.
    - Otherwise append "_exp{experiment_id}" before extension.
    """
    p = Path(base_output)
    suffix = p.suffix
    stem = p.stem if suffix else p.name

    # Update seed token if present (keep filename truthful to the actual seed).
    if re.search(r"(?:^|[_-])seed\d+(?:$|[_-])", stem):
        stem = re.sub(r"seed\d+", f"seed{seed}", stem)

    # Replace existing exp token (exp123, _exp123, -exp123) or append a new one.
    if re.search(r"(?:^|[_-])exp\d+(?:$|[_-])", stem):
        stem = re.sub(r"exp\d+", f"exp{experiment_id}", stem)
    else:
        stem = f"{stem}_exp{experiment_id}"

    if suffix:
        return str(p.with_name(stem + suffix))
    return str(p.with_name(stem))


def configure_torch_runtime(args):
    device = getattr(args, "device", None)
    use_cuda_runtime = getattr(device, "type", None) == "cuda" and torch.cuda.is_available()

    cudnn_benchmark = bool(getattr(args, "cudnn_benchmark", False)) and use_cuda_runtime
    allow_tf32 = bool(getattr(args, "allow_tf32", False)) and use_cuda_runtime

    torch.backends.cudnn.benchmark = cudnn_benchmark
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


def fl_run(args):
    """
    function to run federated learning logics
    """
    # setup logger
    args.logger = setup_logger(
        __name__, f'{args.output}', level=logging.INFO,
        stream=args.log_stream, use_tqdm=args.log_stream,
        color=getattr(args, "log_color", "auto"))
    flush_bootstrap_logs(args, args.logger)
    print_filtered_args(args, args.logger)
    runtime_profiler = RuntimeProfiler(
        args, args.logger, args.output) if args.record_time else None
    configure_torch_runtime(args)
    if runtime_profiler is not None:
        runtime_profiler.log_system_info()
    start_time = time.time()
    args.logger.info(
        f"Started on {time.asctime(time.localtime(start_time))}")
    # fix randomness
    setup_seed(args.seed)

    # 1. load dataset and split dataset indices for clients with i.i.d or non-i.i.d
    train_dataset, test_dataset = load_data(args)
    client_indices, test_dataset = split_dataset(
        args, train_dataset, test_dataset)
    args.logger.info("Data partitioned")

    # 2. initialize clients and server with seperate training data indices
    clients = coordinator.init_clients(
        args, client_indices, train_dataset, test_dataset)
    the_server = Server(args, clients, test_dataset, train_dataset)
    if runtime_profiler is not None:
        runtime_profiler.attach(the_server, clients)

    # 3. initialize the federated learning algorithm for clients and server
    coordinator.set_fl_algorithm(args, the_server, clients)
    args.logger.info("Clients and server are initialized")
    args.logger.info("Starting Training...")
    eval_interval = int(args.eval_interval)
    if eval_interval <= 0:
        args.logger.info(
            "Evaluation schedule | disabled"
        )
    else:
        args.logger.info(
            "Evaluation schedule | interval=%d rounds | final_round_always_eval=True",
            eval_interval,
        )
    torch_profiler = None
    with create_torch_profiler(args, args.output) as torch_profiler:
        for global_epoch in tqdm(range(args.epochs), desc="Global Epochs", dynamic_ncols=True):
            round_start = time.perf_counter()
            if runtime_profiler is not None:
                runtime_profiler.start_round(global_epoch)

            epoch_msg = f"Epoch {global_epoch:<3}\t"
            # print(f"Global epoch {global_epoch} begin")
            # server dispatches numpy version global weights 1d vector to clients
            global_weights_vec = the_server.global_weights_vec

            # clients' local training
            round_train_correct = 0.0
            round_train_loss_sum = 0.0
            round_train_samples = 0
            for client in clients:
                client.load_global_model(global_weights_vec)
                train_acc, train_loss, train_samples = client.local_training()
                client.fetch_updates()
                round_train_correct += train_acc * train_samples
                round_train_loss_sum += train_loss * train_samples
                round_train_samples += train_samples

            avg_train_acc = (
                round_train_correct / round_train_samples
                if round_train_samples > 0 else 0.0
            )
            avg_train_loss = (
                round_train_loss_sum / round_train_samples
                if round_train_samples > 0 else 0.0
            )
            epoch_msg += (
                f"\tTrain Acc: {avg_train_acc:.4f}\tTrain loss: {avg_train_loss:.4f}"
                f"\tTrain samples: {round_train_samples}\t"
            )

            # perform post-training attacks, for omniscient model poisoning attack, pass all clients
            omniscient_attack(clients)

            # server collects weights from clients
            the_server.collect_updates(global_epoch)
            the_server.aggregation()
            the_server.update_global()

            # evalute the attack success rate (ASR) when a backdoor attack is launched
            should_eval = should_run_evaluation(
                global_epoch=global_epoch,
                total_epochs=args.epochs,
                eval_interval=eval_interval,
            )
            test_stats = {}
            eval_start = time.perf_counter()
            if should_eval:
                test_stats = coordinator.evaluate(
                    the_server, test_dataset, args, global_epoch)
            if runtime_profiler is not None and should_eval:
                runtime_profiler.add_server_stage(
                    "evaluation", time.perf_counter() - eval_start)

            # print the training and testing results of the current global_epoch
            log_start = time.perf_counter()
            round_time_sec = time.perf_counter() - round_start
            epoch_msg += f"Round time: {round_time_sec:.2f}s\t"
            if test_stats:
                epoch_msg += "\t".join(
                    [f"{key}: {value:.4f}" for key, value in test_stats.items()])
            args.logger.info(epoch_msg)
            if runtime_profiler is not None:
                runtime_profiler.add_server_stage(
                    "logging", time.perf_counter() - log_start)
                runtime_profiler.finish_round(
                    total_sec=time.perf_counter() - round_start,
                    train_acc=avg_train_acc,
                    train_loss=avg_train_loss,
                    train_samples=round_train_samples,
                    test_stats=test_stats,
                )
            if torch_profiler is not None:
                torch_profiler.step()

            # clear memory (low-frequency to reduce overhead)
            if (global_epoch + 1) % 20 == 0:
                gc.collect()

    if args.record_time:
        report_time(clients, the_server)
        runtime_profiler.finalize()
    if torch_profiler is not None:
        summarize_torch_profiler(torch_profiler, args.logger, args.device)

    plot_accuracy(args.output)

    end_time = time.time()
    time_difference = end_time - start_time
    minutes, seconds = int(
        time_difference // 60), int(time_difference % 60)
    args.logger.info(
        f"Training finished on {time.asctime(time.localtime(end_time))} using {minutes} minutes and {seconds} seconds in total.")


def report_time(clients, the_server):
    [c.time_recorder.report(f"Client {idx}") for idx, c in enumerate(clients)]
    the_server.time_recorder.report("Server")


def omniscient_attack(clients):
    """
    Perform an omniscient attack, which involves eavesdropping or collusion
    between malicious clients to craft adversarial updates.
    """
    # Filter out all omniscient attackers from the client list
    omniscient_attackers = [
        client for client in clients
        if client.category == "attacker" and "omniscient" in client.attributes
    ]

    # If no omniscient attackers exist, exit early
    if not omniscient_attackers:
        return
    # Generate malicious updates using the first attacker's logic
    malicious_updates = omniscient_attackers[0].omniscient(clients)
    if malicious_updates is None:
        raise ValueError("No updates generated by the omniscient attacker")

    if getattr(omniscient_attackers[0], "shared_omniscient_update", False):
        shared_update = malicious_updates
        if getattr(shared_update, "ndim", 0) > 1 and shared_update.shape[0] == 1:
            shared_update = shared_update[0]
        for client in omniscient_attackers:
            client.update = shared_update
        return

    # Check if the malicious update is a single vector or a batch of updates
    is_single_update = len(
        malicious_updates.shape) == 1 or malicious_updates.shape[0] == 1

    if is_single_update:
        # If a single update is provided, all attackers perform their own attack
        omniscient_attackers[0].update = malicious_updates
        for client in omniscient_attackers[1:]:
            client.update = client.omniscient(clients)
    else:
        # If multiple updates are provided, assign each update to an attacker
        # An attack method aiming to provide the same updates for all attackers can return repeated updates.
        for client, update in zip(omniscient_attackers, malicious_updates):
            client.update = update


def main(args, cli_args):
    """
    preprocess the arguments, logics, and run the federated learning process
    """
    # if Benchmarks is True, run all combinations of attacks and defenses
    if cli_args.benchmark:
        benchmark_preprocess(args)
        fl_run(args)
    else:
        override_args(args, cli_args)
        single_preprocess(args)
        # Repeat experiments with deterministic seeding:
        # seed starts from args.seed (after CLI/YAML override) and increments by 1 for each experiment.
        # Prefer CLI if provided; otherwise fall back to YAML (args.*); else defaults.
        num_experiments = (
            cli_args.num_experiments
            if cli_args.num_experiments is not None
            else getattr(args, "num_experiments", None)
        )
        start_experiment_id = (
            cli_args.experiment_id
            if cli_args.experiment_id is not None
            else getattr(args, "experiment_id", None)
        )
        num_experiments = 1 if num_experiments is None else int(num_experiments)
        start_experiment_id = 0 if start_experiment_id is None else int(start_experiment_id)

        if not hasattr(args, "seed") or args.seed is None:
            raise ValueError("Missing seed. Set `seed` in YAML or pass `--seed` on CLI.")

        seed_start = int(args.seed)
        base_output = args.output

        for exp_offset in range(int(num_experiments)):
            exp_id = int(start_experiment_id) + exp_offset
            seed = seed_start + exp_id

            args.experiment_id = exp_id
            args.seed_start = seed_start
            args.seed = seed

            if base_output:
                args.output = _output_with_experiment_id(base_output, seed, exp_id)

            fl_run(args)

            # best-effort cleanup between experiments to reduce memory pressure
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
