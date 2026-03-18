import argparse
import logging
import os
import subprocess
import sys
from functools import partial
from multiprocessing import Pool

from flpoison.fl.configuration import read_yaml
from flpoison.utils.config_utils import preset_relpath
from flpoison.utils.global_utils import setup_console_logger


LOGGER = setup_console_logger("flpoison.batchrun", level=logging.INFO)


def run_command(command, file_name):
    # Check if the old file exists
    if os.path.exists(f"{file_name}"):
        LOGGER.info("File %s exists, skip", file_name)
        return

    LOGGER.info("Running command: %s", command)
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # Define log files for stdout and stderr
    tmp = file_name.replace("logs", "err_logs")
    out_error_file = f"{tmp[:-4]}.err"

    process = subprocess.Popen(
        command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pid = process.pid
    LOGGER.info("Started command with PID: %d", pid)

    stdout, stderr = process.communicate()
    if process.returncode == 0:
        LOGGER.info("Command %s finished successfully with PID: %d", command, pid)
    else:
        LOGGER.error("Command %s failed with PID: %d", command, pid)
        if stderr:
            LOGGER.error("Error: %s", stderr.strip())
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with open(out_error_file, 'w') as out_error_log:
            out_error_log.write(stdout)  # Save stdout to the log file
            out_error_log.write(stderr)  # Save stderr to the error file


def get_configs(dataset, algorithm, distribution, defense):
    params = {
        "MNIST": {
            "FedSGD": {"epoch": 300, "lr": 0.01},
            "FedOpt": {"epoch": 100, "lr": 0.01}
        },
        "CIFAR10": {
            "FedSGD": {
                "epoch": 300, "lr": 0.05,
                "non-iid": {
                    "defenses": ["Krum", "MultiKrum", "Bucketing", "Bulyan", "SignGuard", "DnC", "FLAME"],
                    "lr": 0.002
                }
            },
            "FedOpt": {
                "epoch": 300, "lr": 0.02,
                "non-iid": {
                    "defenses": ["Krum", "Bucketing"],
                    "lr": 0.002
                }
            }
        },
        "TinyImageNet": {
            "FedSGD": {"epoch": 150, "lr": 0.05}
        },
        "CHMNIST": {
            "FedSGD": {"epoch": 150, "lr": 0.001}
        },
    }

    dataset_params = params.get(dataset, {})
    num_clients = 20 if dataset == "CIFAR10" else 50
    algo_params = dataset_params.get(algorithm, {})

    if isinstance(algo_params, dict):
        epoch = algo_params["epoch"]
        lr = algo_params["lr"]

        # Check for non-iid specific overrides
        if distribution == "non-iid" and "non-iid" in algo_params:
            non_iid_params = algo_params["non-iid"]
            if defense in non_iid_params.get("defenses", []):
                lr = non_iid_params.get("lr", lr)

        return num_clients, epoch, lr

    raise ValueError(f"Invalid configuration for {dataset} with {algorithm}")


def main(args):
    distributions = ['iid', 'non-iid', 'class-imbalanced_iid']
    algorithms = ['FedSGD', 'FedAvg', 'FedOpt']
    folder_name = 'FLPoison'
    gpu_idx = 1
    MAX_PROCESSES = 6

    # set them from the arguments
    dataset = args.dataset
    model = args.model
    attacks = args.attacks
    defenses = args.defenses
    distributions = args.distributions
    algorithms = args.algorithms
    gpu_idx = args.gpu_idx
    MAX_PROCESSES = args.max_processes
    datasets_models = [(dataset, model)]

    # check folder
    current_dir = os.getcwd()
    if folder_name in current_dir:
        repo_dir = current_dir
    elif os.path.isdir(os.path.join(current_dir, folder_name)):
        repo_dir = os.path.join(current_dir, folder_name)
    else:
        LOGGER.error(
            "The current directory '%s' is not in %s and does not contain a %s folder.",
            current_dir,
            folder_name,
            folder_name,
        )
        sys.exit(1)

    # Define the pool
    pool = Pool(processes=MAX_PROCESSES)
    tasks = []
    for algorithm in algorithms:
        for dataset, model in datasets_models:
            config_file = preset_relpath(algorithm, dataset, model).as_posix()
            for distribution in distributions:
                for attack in attacks:
                    for defense in defenses:
                        num_clients, epoch, learning_rate = get_configs(
                            dataset, algorithm, distribution, defense)

                        command = f'python -u -m flpoison -config=./{config_file} -data {dataset} -model {model} -e {epoch} -att {attack} -def {defense} -dtb {distribution} -alg {algorithm} -lr {learning_rate} -gidx {gpu_idx}'
                        file_name = f'{repo_dir}/logs/{algorithm}/{dataset}_{model}/{distribution}/{dataset}_{model}_{distribution}_{attack}_{defense}_{epoch}_{num_clients}_{learning_rate}_{algorithm}.txt'

                        # Add the task to the list
                        tasks.append((command, file_name))

    # Use pool.map to run the commands in parallel
    pool.starmap(partial(run_command), tasks)

    # Close and wait for the pool to finish
    pool.close()
    pool.join()


def get_all_attacks_defenses():
    args = vars(read_yaml(preset_relpath('FedSGD', 'MNIST', 'lenet').as_posix()))
    attacks = [attack_i['attack'] for attack_i in args['attacks']]
    defenses = [defense_j['defense'] for defense_j in args['defenses']]
    return attacks, defenses


def test():
    attacks, defenses = get_all_attacks_defenses()
    LOGGER.info("attacks = %s\n", attacks)
    LOGGER.info("defenses = %s", defenses)


def main_entry():
    parser = argparse.ArgumentParser(
        description="Run distributed training with attacks and defenses."
    )

    parser.add_argument(
        '-distributions',
        '--distributions',
        nargs='+',
        default=['iid', 'non-iid'],
        help="List of distributions to use. Default is ['iid'].",
    )
    parser.add_argument(
        '-algorithms',
        '--algorithms',
        nargs='+',
        default=['FedSGD', 'FedOpt'],
        help="List of algorithm types to use. Default is ['FedSGD'].",
    )
    parser.add_argument('-data', '--dataset', type=str, default='MNIST', help="Dataset to use. Default is MNIST.")
    parser.add_argument('-model', '--model', type=str, default='lenet', help="Model to use. Default is lenet.")
    parser.add_argument('-gidx', '--gpu_idx', type=int, default=1, help="GPU index to use. Default is 1.")
    parser.add_argument('-maxp', '--max_processes', type=int, default=6, help="Max number of process parallel. Default is 6.")
    parser.add_argument(
        '-attacks',
        '--attacks',
        nargs='+',
        default=['NoAttack', 'Gaussian', 'SignFlipping', 'IPM', 'ALIE', 'FangAttack', 'MinMax', 'MinSum', 'Mimic', 'LabelFlipping', 'BadNets', 'ModelReplacement', 'DBA', 'AlterMin', 'EdgeCase', 'Neurotoxin'],
        help="List of attacks to use.",
    )
    parser.add_argument(
        '-defenses',
        '--defenses',
        nargs='+',
        default=['Mean', 'SimpleClustering', 'Krum', 'MultiKrum', 'TrimmedMean', 'Median', 'Bulyan', 'RFA', 'FLTrust', 'CenteredClipping', 'DnC', 'Bucketing', 'SignGuard', 'TriGuardFL', 'LASA', 'Auror', 'FoolsGold', 'NormClipping', 'CRFL', 'DeepSight', 'FLAME'],
        help="List of defenses to use.",
    )
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    main_entry()
