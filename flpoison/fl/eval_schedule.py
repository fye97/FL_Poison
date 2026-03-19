DEFAULT_EVAL_INTERVAL = 0


def should_run_evaluation(global_epoch: int, total_epochs: int, eval_interval: int) -> bool:
    """
    Run full evaluation every ``eval_interval`` rounds and always on the final
    round when evaluation is enabled. Non-positive intervals disable evaluation.
    """
    if total_epochs <= 0:
        return False
    interval = int(eval_interval)
    if interval <= 0:
        return False
    return global_epoch == total_epochs - 1 or (global_epoch + 1) % interval == 0
