import json
import os
import platform
import shutil
import subprocess
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import torch


CLIENT_STAGE_NAMES = (
    "sync",
    "data",
    "fwd_bwd",
    "gpu_compute",
    "opt_step",
    "pack_update",
)
ROUND_STAGE_NAMES = (
    "sync",
    "data",
    "fwd_bwd",
    "gpu_compute",
    "opt_step",
    "pack_update",
    "collect_updates",
    "aggregate",
    "defense",
    "evaluation",
    "logging",
)


def _derived_output_path(output_file, subdir, suffix):
    output_file = str(output_file)
    marker = f"logs{os.sep}"
    replacement = f"logs{os.sep}{subdir}{os.sep}"
    if marker in output_file:
        derived = output_file.replace(marker, replacement, 1)
        path = Path(derived)
    else:
        source = Path(output_file)
        path = source.parent / subdir / source.name
    path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def perf_summary_path(output_file):
    return _derived_output_path(output_file, "perf_logs", ".json")


def torch_trace_dir(output_file):
    trace_path = _derived_output_path(output_file, "torch_traces", "")
    trace_path.mkdir(parents=True, exist_ok=True)
    return trace_path


def _mean(values):
    return sum(values) / len(values) if values else None


def _fmt_float(value, digits=2, suffix=""):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


class _GPUSampler:
    def __init__(self, device, interval_ms=100):
        self.device = device
        self.device_index = self._resolve_device_index(device)
        self.interval_sec = max(0.01, float(interval_ms) / 1000.0)
        self.samples = []
        self._stop_event = threading.Event()
        self._thread = None
        self._nvidia_smi = shutil.which("nvidia-smi")

    def _resolve_device_index(self, device):
        if getattr(device, "index", None) is not None:
            return int(device.index)
        try:
            return int(torch.cuda.current_device())
        except Exception:
            return 0

    def _safe_cuda_call(self, fn):
        try:
            return fn(self.device)
        except Exception:
            return None

    def _nvidia_smi_sample(self):
        if not self._nvidia_smi:
            return {}
        cmd = [
            self._nvidia_smi,
            f"--id={self.device_index}",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2.0,
            )
        except Exception:
            return {}
        line = result.stdout.strip().splitlines()
        if not line:
            return {}
        fields = [field.strip() for field in line[0].split(",")]
        if len(fields) != 4:
            return {}
        try:
            used_mb = float(fields[2])
            total_mb = float(fields[3])
            memory_util = (used_mb / total_mb * 100.0) if total_mb > 0 else None
            return {
                "utilization_pct": float(fields[0]),
                "memory_utilization_pct": float(fields[1]) if fields[1] not in {"", "[Not Supported]"} else memory_util,
                "memory_used_mb": used_mb,
                "memory_total_mb": total_mb,
            }
        except Exception:
            return {}

    def _read_sample(self):
        sample = {
            "timestamp": time.perf_counter(),
            "utilization_pct": None,
            "memory_utilization_pct": None,
            "memory_allocated_mb": None,
            "memory_reserved_mb": None,
            "max_memory_allocated_mb": None,
            "max_memory_reserved_mb": None,
            "memory_used_mb": None,
            "memory_total_mb": None,
        }

        utilization = self._safe_cuda_call(torch.cuda.utilization)
        if utilization is not None:
            sample["utilization_pct"] = float(utilization)
        memory_usage = self._safe_cuda_call(torch.cuda.memory_usage)
        if memory_usage is not None:
            sample["memory_utilization_pct"] = float(memory_usage)

        memory_allocated = self._safe_cuda_call(torch.cuda.memory_allocated)
        if memory_allocated is not None:
            sample["memory_allocated_mb"] = float(memory_allocated / (1024 ** 2))
        memory_reserved = self._safe_cuda_call(torch.cuda.memory_reserved)
        if memory_reserved is not None:
            sample["memory_reserved_mb"] = float(memory_reserved / (1024 ** 2))
        max_memory_allocated = self._safe_cuda_call(torch.cuda.max_memory_allocated)
        if max_memory_allocated is not None:
            sample["max_memory_allocated_mb"] = float(max_memory_allocated / (1024 ** 2))
        max_memory_reserved = self._safe_cuda_call(torch.cuda.max_memory_reserved)
        if max_memory_reserved is not None:
            sample["max_memory_reserved_mb"] = float(max_memory_reserved / (1024 ** 2))

        if sample["utilization_pct"] is None or sample["memory_utilization_pct"] is None:
            nvidia_smi_sample = self._nvidia_smi_sample()
            for key, value in nvidia_smi_sample.items():
                if sample.get(key) is None:
                    sample[key] = value

        has_signal = any(
            sample[key] is not None
            for key in (
                "utilization_pct",
                "memory_utilization_pct",
                "memory_allocated_mb",
                "memory_reserved_mb",
                "max_memory_allocated_mb",
                "max_memory_reserved_mb",
                "memory_used_mb",
            )
        )
        if not has_signal:
            return None
        return sample

    def _run(self):
        while not self._stop_event.is_set():
            sample = self._read_sample()
            if sample is not None:
                self.samples.append(sample)
            self._stop_event.wait(self.interval_sec)

    def start(self):
        self.samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return []
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        return list(self.samples)


class RuntimeProfiler:
    def __init__(self, args, logger, output_file):
        self.args = args
        self.logger = logger
        self.output_file = output_file
        self.device = getattr(args, "device", torch.device("cpu"))
        self.summary_path = perf_summary_path(output_file)
        self.trace_dir = torch_trace_dir(output_file)
        self.rounds = []
        self._current_round = None
        self._aggregation_breakdown = None
        self._gpu_sampler = None
        self.system_info = self._collect_system_info()

    def _collect_system_info(self):
        info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "pytorch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device": str(self.device),
            "device_type": getattr(self.device, "type", "cpu"),
            "gpu_name": None,
            "gpu_count": 0,
            "gpu_total_memory_mb": None,
        }
        if getattr(self.device, "type", None) == "cuda" and torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(self.device)
                info["gpu_name"] = props.name
                info["gpu_count"] = torch.cuda.device_count()
                info["gpu_total_memory_mb"] = float(props.total_memory / (1024 ** 2))
            except Exception:
                pass
        return info

    def _experiment_metadata(self):
        return {
            "seed": getattr(self.args, "seed", None),
            "dataset": getattr(self.args, "dataset", None),
            "model": getattr(self.args, "model", None),
            "algorithm": getattr(self.args, "algorithm", None),
            "attack": getattr(self.args, "attack", None),
            "defense": getattr(self.args, "defense", None),
            "batch_size": getattr(self.args, "batch_size", None),
            "eval_batch_size": getattr(self.args, "eval_batch_size", None),
            "num_clients": getattr(self.args, "num_clients", None),
            "num_adv": getattr(self.args, "num_adv", None),
            "local_epochs": getattr(self.args, "local_epochs", None),
            "epochs": getattr(self.args, "epochs", None),
            "distribution": getattr(self.args, "distribution", None),
            "dirichlet_alpha": getattr(self.args, "dirichlet_alpha", None),
            "cache_partition": getattr(self.args, "cache_partition", None),
            "num_workers": getattr(self.args, "num_workers", None),
            "eval_interval": getattr(self.args, "eval_interval", None),
            "gpu_sample_interval_ms": getattr(self.args, "gpu_sample_interval_ms", None),
            "torch_profile": bool(getattr(self.args, "torch_profile", False)),
            "output": str(self.output_file),
        }

    def attach(self, server, clients):
        server.runtime_profiler = self
        if hasattr(server, "aggregator") and hasattr(server.aggregator, "bind_runtime_profiler"):
            server.aggregator.bind_runtime_profiler(self)
        for client in clients:
            client.runtime_profiler = self

    def log_system_info(self):
        meta = self._experiment_metadata()
        self.logger.info(
            "Performance baseline | "
            f"seed={meta['seed']} dataset={meta['dataset']} model={meta['model']} "
            f"algorithm={meta['algorithm']} defense={meta['defense']} "
            f"batch_size={meta['batch_size']} clients={meta['num_clients']} "
            f"local_epochs={meta['local_epochs']} distribution={meta['distribution']}"
        )
        self.logger.info(
            "System info | "
            f"device={self.system_info['device']} gpu={self.system_info['gpu_name'] or 'n/a'} "
            f"torch={self.system_info['pytorch_version']} cuda={self.system_info['torch_cuda_version'] or 'n/a'} "
            f"cudnn={self.system_info['cudnn_version'] or 'n/a'}"
        )

    def start_round(self, round_idx):
        if self._current_round is not None:
            raise RuntimeError("Cannot start a new round before finishing the current round.")

        self._current_round = {
            "round": int(round_idx),
            "stage_times": {name: 0.0 for name in ROUND_STAGE_NAMES},
            "clients": {},
            "server_internal_clients": {},
            "gpu": {
                "samples": [],
                "utilization_pct_avg": None,
                "memory_utilization_pct_avg": None,
                "memory_allocated_mb_avg": None,
                "memory_reserved_mb_avg": None,
                "memory_peak_allocated_mb": None,
                "memory_peak_reserved_mb": None,
            },
            "train_metrics": {},
            "test_metrics": {},
            "sec_per_client": None,
            "rounds_per_sec": None,
            "gpu_compute_ratio": None,
            "aggregation_split_available": False,
            "round_time_sec": 0.0,
            "total_sec": 0.0,
        }
        self._aggregation_breakdown = None

        if getattr(self.device, "type", None) == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
                self._gpu_sampler = _GPUSampler(
                    self.device, interval_ms=getattr(self.args, "gpu_sample_interval_ms", 100)
                )
                self._gpu_sampler.start()
            except Exception:
                self._gpu_sampler = None

    def add_client_stage(self, worker_id, stage, duration):
        if self._current_round is None:
            return
        duration = float(duration)
        bucket_name = "clients" if worker_id >= 0 else "server_internal_clients"
        bucket = self._current_round[bucket_name].setdefault(str(worker_id), {})
        bucket[stage] = bucket.get(stage, 0.0) + duration
        if worker_id >= 0:
            self._current_round["stage_times"][stage] += duration

    def add_server_stage(self, stage, duration):
        if self._current_round is None:
            return
        self._current_round["stage_times"][stage] += float(duration)

    def begin_aggregation_breakdown(self):
        self._aggregation_breakdown = {"aggregate": 0.0, "defense": 0.0}

    def add_aggregation_substage(self, stage, duration):
        if self._aggregation_breakdown is None:
            return
        if stage not in self._aggregation_breakdown:
            raise ValueError(f"Unsupported aggregation substage: {stage}")
        self._aggregation_breakdown[stage] += float(duration)

    def finish_aggregation(self, total_duration):
        if self._current_round is None:
            return

        total_duration = float(total_duration)
        aggregate_time = 0.0
        defense_time = 0.0
        if self._aggregation_breakdown is not None:
            aggregate_time = self._aggregation_breakdown.get("aggregate", 0.0)
            defense_time = self._aggregation_breakdown.get("defense", 0.0)
            self._current_round["aggregation_split_available"] = (aggregate_time + defense_time) > 0.0

        accounted = aggregate_time + defense_time
        remainder = max(0.0, total_duration - accounted)
        self._current_round["stage_times"]["aggregate"] += aggregate_time + remainder
        self._current_round["stage_times"]["defense"] += defense_time
        self._aggregation_breakdown = None

    def finish_round(self, total_sec, train_acc, train_loss, train_samples=0, test_stats=None):
        if self._current_round is None:
            raise RuntimeError("No active round to finish.")

        total_sec = float(total_sec)
        self._current_round["round_time_sec"] = total_sec
        self._current_round["total_sec"] = total_sec
        self._current_round["train_metrics"] = {
            "train_acc": float(train_acc),
            "train_loss": float(train_loss),
            "train_samples": int(train_samples),
        }
        self._current_round["test_metrics"] = {
            key: float(value) for key, value in (test_stats or {}).items()
        }

        samples = self._gpu_sampler.stop() if self._gpu_sampler is not None else []
        self._gpu_sampler = None
        if getattr(self.device, "type", None) == "cuda" and torch.cuda.is_available():
            try:
                self._current_round["gpu"]["memory_peak_allocated_mb"] = float(
                    torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                )
                self._current_round["gpu"]["memory_peak_reserved_mb"] = float(
                    torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
                )
            except Exception:
                pass
        if samples:
            self._current_round["gpu"]["samples"] = samples
            self._current_round["gpu"]["utilization_pct_avg"] = _mean(
                [sample["utilization_pct"] for sample in samples if sample["utilization_pct"] is not None]
            )
            self._current_round["gpu"]["memory_utilization_pct_avg"] = _mean(
                [sample["memory_utilization_pct"] for sample in samples if sample["memory_utilization_pct"] is not None]
            )
            self._current_round["gpu"]["memory_allocated_mb_avg"] = _mean(
                [
                    sample["memory_allocated_mb"]
                    for sample in samples
                    if sample["memory_allocated_mb"] is not None
                ]
            )
            self._current_round["gpu"]["memory_reserved_mb_avg"] = _mean(
                [
                    sample["memory_reserved_mb"]
                    for sample in samples
                    if sample["memory_reserved_mb"] is not None
                ]
            )
            if self._current_round["gpu"]["memory_allocated_mb_avg"] is None:
                self._current_round["gpu"]["memory_allocated_mb_avg"] = _mean(
                    [sample["memory_used_mb"] for sample in samples if sample["memory_used_mb"] is not None]
                )
            if self._current_round["gpu"]["memory_peak_allocated_mb"] is None:
                peak_allocated = [
                    sample["max_memory_allocated_mb"]
                    for sample in samples
                    if sample["max_memory_allocated_mb"] is not None
                ]
                if peak_allocated:
                    self._current_round["gpu"]["memory_peak_allocated_mb"] = max(peak_allocated)
            if self._current_round["gpu"]["memory_peak_reserved_mb"] is None:
                peak_reserved = [
                    sample["max_memory_reserved_mb"]
                    for sample in samples
                    if sample["max_memory_reserved_mb"] is not None
                ]
                if peak_reserved:
                    self._current_round["gpu"]["memory_peak_reserved_mb"] = max(peak_reserved)

        client_totals = []
        for client_metrics in self._current_round["clients"].values():
            total_client_time = sum(client_metrics.get(stage, 0.0) for stage in CLIENT_STAGE_NAMES)
            client_metrics["total_sec"] = total_client_time
            client_totals.append(total_client_time)
        for client_metrics in self._current_round["server_internal_clients"].values():
            client_metrics["total_sec"] = sum(
                client_metrics.get(stage, 0.0) for stage in CLIENT_STAGE_NAMES
            )

        self._current_round["sec_per_client"] = _mean(client_totals)
        if total_sec > 0:
            self._current_round["rounds_per_sec"] = 1.0 / float(total_sec)
            self._current_round["gpu_compute_ratio"] = (
                self._current_round["stage_times"]["gpu_compute"] / float(total_sec)
            )

        self.rounds.append(self._current_round)
        self.logger.info(self._format_round_summary(self._current_round))
        self._current_round = None

    def _format_round_summary(self, round_record):
        lines = [f"Round {round_record['round']} summary"]
        lines.append(f"total: {_fmt_float(round_record['total_sec'], digits=2, suffix='s')}")
        lines.append(
            f"sync: {_fmt_float(round_record['stage_times']['sync'], digits=2, suffix='s')}"
        )
        lines.append(
            f"data: {_fmt_float(round_record['stage_times']['data'], digits=2, suffix='s')}"
        )
        lines.append(
            f"fwd_bwd: {_fmt_float(round_record['stage_times']['fwd_bwd'], digits=2, suffix='s')}"
        )
        lines.append(
            f"gpu_compute: {_fmt_float(round_record['stage_times']['gpu_compute'], digits=2, suffix='s')}"
        )
        lines.append(
            f"opt_step: {_fmt_float(round_record['stage_times']['opt_step'], digits=2, suffix='s')}"
        )
        lines.append(
            f"pack_update: {_fmt_float(round_record['stage_times']['pack_update'], digits=2, suffix='s')}"
        )
        lines.append(
            f"aggregate: {_fmt_float(round_record['stage_times']['aggregate'], digits=2, suffix='s')}"
        )
        lines.append(
            f"defense: {_fmt_float(round_record['stage_times']['defense'], digits=2, suffix='s')}"
        )
        if round_record["stage_times"]["collect_updates"] > 0:
            lines.append(
                f"collect_updates: {_fmt_float(round_record['stage_times']['collect_updates'], digits=2, suffix='s')}"
            )
        if round_record["stage_times"]["evaluation"] > 0:
            lines.append(
                f"evaluation: {_fmt_float(round_record['stage_times']['evaluation'], digits=2, suffix='s')}"
            )
        lines.append(
            f"logging: {_fmt_float(round_record['stage_times']['logging'], digits=2, suffix='s')}"
        )
        lines.append(
            "sec/client: "
            f"{_fmt_float(round_record['sec_per_client'], digits=3, suffix='s')}"
        )
        lines.append(
            "gpu util avg: "
            f"{_fmt_float(round_record['gpu']['utilization_pct_avg'], digits=1, suffix='%')}"
        )
        lines.append(
            "gpu compute/round: "
            f"{_fmt_float((round_record['gpu_compute_ratio'] or 0.0) * 100.0 if round_record['gpu_compute_ratio'] is not None else None, digits=1, suffix='%')}"
        )
        lines.append(
            "gpu mem avg: "
            f"{_fmt_float(round_record['gpu']['memory_allocated_mb_avg'], digits=1, suffix='MB')}"
        )
        lines.append(
            "gpu mem peak: "
            f"{_fmt_float(round_record['gpu']['memory_peak_allocated_mb'], digits=1, suffix='MB')}"
        )
        lines.append(
            "train accuracy: "
            f"{_fmt_float(round_record['train_metrics'].get('train_acc'), digits=4)}"
        )
        lines.append(
            "train samples: "
            f"{round_record['train_metrics'].get('train_samples', 0)}"
        )
        if "Test Acc" in round_record["test_metrics"]:
            lines.append(
                "val accuracy: "
                f"{_fmt_float(round_record['test_metrics'].get('Test Acc'), digits=4)}"
            )
        return "\n".join(lines)

    def _build_overall_summary(self):
        total_rounds = len(self.rounds)
        total_time = sum(round_record["total_sec"] for round_record in self.rounds)
        sec_per_round = (total_time / total_rounds) if total_rounds else None
        rounds_per_sec = (total_rounds / total_time) if total_time > 0 else None
        sec_per_client = _mean(
            [round_record["sec_per_client"] for round_record in self.rounds if round_record["sec_per_client"] is not None]
        )
        gpu_util = _mean(
            [
                round_record["gpu"]["utilization_pct_avg"]
                for round_record in self.rounds
                if round_record["gpu"]["utilization_pct_avg"] is not None
            ]
        )
        gpu_compute_ratio = _mean(
            [
                round_record["gpu_compute_ratio"]
                for round_record in self.rounds
                if round_record["gpu_compute_ratio"] is not None
            ]
        )
        avg_mem = _mean(
            [
                round_record["gpu"]["memory_allocated_mb_avg"]
                for round_record in self.rounds
                if round_record["gpu"]["memory_allocated_mb_avg"] is not None
            ]
        )
        peak_mem_candidates = [
            round_record["gpu"]["memory_peak_allocated_mb"]
            for round_record in self.rounds
            if round_record["gpu"]["memory_peak_allocated_mb"] is not None
        ]
        peak_mem = max(peak_mem_candidates) if peak_mem_candidates else None
        final_round = self.rounds[-1] if self.rounds else None

        return {
            "num_rounds": total_rounds,
            "total_time_sec": total_time,
            "sec_per_round": sec_per_round,
            "rounds_per_sec": rounds_per_sec,
            "sec_per_client": sec_per_client,
            "gpu_utilization_pct_avg": gpu_util,
            "gpu_compute_ratio_avg": gpu_compute_ratio,
            "gpu_memory_allocated_mb_avg": avg_mem,
            "gpu_memory_peak_allocated_mb": peak_mem,
            "stage_time_sec_avg": {
                stage: _mean([round_record["stage_times"][stage] for round_record in self.rounds])
                for stage in ROUND_STAGE_NAMES
            },
            "final_train_accuracy": (
                final_round["train_metrics"].get("train_acc") if final_round is not None else None
            ),
            "final_train_loss": (
                final_round["train_metrics"].get("train_loss") if final_round is not None else None
            ),
            "final_train_samples": (
                final_round["train_metrics"].get("train_samples") if final_round is not None else None
            ),
            "final_test_metrics": (
                final_round["test_metrics"] if final_round is not None else {}
            ),
        }

    def finalize(self):
        payload = {
            "metadata": self._experiment_metadata(),
            "system_info": self.system_info,
            "overall": self._build_overall_summary(),
            "rounds": self.rounds,
        }
        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        self.logger.info(
            "Performance summary | "
            f"sec/round={_fmt_float(payload['overall']['sec_per_round'], digits=3, suffix='s')} "
            f"rounds/sec={_fmt_float(payload['overall']['rounds_per_sec'], digits=3)} "
            f"sec/client={_fmt_float(payload['overall']['sec_per_client'], digits=3, suffix='s')} "
            f"gpu_util={_fmt_float(payload['overall']['gpu_utilization_pct_avg'], digits=1, suffix='%')} "
            f"gpu_mem_peak={_fmt_float(payload['overall']['gpu_memory_peak_allocated_mb'], digits=1, suffix='MB')} "
            f"train_acc={_fmt_float(payload['overall']['final_train_accuracy'], digits=4)}"
        )
        if payload["overall"]["final_test_metrics"]:
            test_summary = ", ".join(
                f"{key}={_fmt_float(value, digits=4)}"
                for key, value in payload["overall"]["final_test_metrics"].items()
            )
            self.logger.info(f"Final test metrics | {test_summary}")
        self.logger.info(f"Performance JSON saved to {self.summary_path}")
        return payload


class CudaEventTimer:
    def __init__(self, device):
        self.device = device
        self.enabled = getattr(device, "type", None) == "cuda" and torch.cuda.is_available()
        self._start_event = None
        self._end_event = None
        self.wall_sec = 0.0
        self.gpu_sec = 0.0
        self._wall_start = None

    def __enter__(self):
        self._wall_start = time.perf_counter()
        if self.enabled:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            self._end_event.record()
            torch.cuda.synchronize(self.device)
            self.gpu_sec = float(self._start_event.elapsed_time(self._end_event) / 1000.0)
        self.wall_sec = float(time.perf_counter() - self._wall_start)
        return False


def create_torch_profiler(args, output_file):
    if not getattr(args, "torch_profile", False):
        return nullcontext()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if getattr(getattr(args, "device", None), "type", None) == "cuda" and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    trace_dir = torch_trace_dir(output_file)
    schedule = torch.profiler.schedule(
        wait=int(getattr(args, "torch_profile_wait", 0) or 0),
        warmup=int(getattr(args, "torch_profile_warmup", 1) or 1),
        active=int(getattr(args, "torch_profile_active", 3) or 3),
        repeat=int(getattr(args, "torch_profile_repeat", 1) or 1),
    )
    return torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
        record_shapes=bool(getattr(args, "torch_profile_record_shapes", True)),
        profile_memory=bool(getattr(args, "torch_profile_memory", True)),
        with_stack=bool(getattr(args, "torch_profile_with_stack", False)),
    )


def summarize_torch_profiler(profiler, logger, device):
    if profiler is None:
        return

    events = profiler.key_averages()
    if not events:
        logger.info("Torch profiler collected no events.")
        return

    sort_key = (
        "self_cuda_time_total"
        if getattr(device, "type", None) == "cuda" and torch.cuda.is_available()
        else "self_cpu_time_total"
    )
    logger.info("Torch profiler top ops:\n%s", events.table(sort_by=sort_key, row_limit=30))

    memcpy_count = 0
    memcpy_cuda_ms = 0.0
    dataloader_cpu_ms = 0.0
    python_cpu_ms = 0.0
    small_kernel_count = 0

    for event in events:
        key = str(getattr(event, "key", ""))
        key_lower = key.lower()
        self_cuda_us = float(getattr(event, "self_cuda_time_total", 0.0) or 0.0)
        self_cpu_us = float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)
        count = int(getattr(event, "count", 0) or 0)

        if "memcpy" in key_lower:
            memcpy_count += count
            memcpy_cuda_ms += self_cuda_us / 1000.0
        if "dataloader" in key_lower:
            dataloader_cpu_ms += self_cpu_us / 1000.0
        if key_lower.startswith("python::") or "[python]" in key_lower:
            python_cpu_ms += self_cpu_us / 1000.0
        if count > 0 and self_cuda_us > 0.0 and (self_cuda_us / count) < 50.0 and count >= 100:
            small_kernel_count += 1

    logger.info(
        "Torch profiler hints | "
        f"cudaMemcpy ops={memcpy_count} cudaMemcpy self time={memcpy_cuda_ms:.2f}ms "
        f"DataLoader self CPU={dataloader_cpu_ms:.2f}ms "
        f"Python self CPU={python_cpu_ms:.2f}ms "
        f"small-kernel hotspots={small_kernel_count}"
    )
