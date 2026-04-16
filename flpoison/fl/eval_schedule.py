DEFAULT_EVALUATE = False


def should_run_evaluation(evaluate: bool) -> bool:
    """
    Evaluation is now controlled by a fixed boolean switch:
    enabled means evaluate every round, disabled means never.
    """
    return bool(evaluate)
