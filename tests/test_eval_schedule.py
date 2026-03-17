import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_schedule import DEFAULT_EVAL_INTERVAL, should_run_evaluation


def test_should_run_evaluation_uses_interval_and_final_round():
    scheduled = [
        epoch for epoch in range(20)
        if should_run_evaluation(epoch, total_epochs=20, eval_interval=5)
    ]
    assert scheduled == [4, 9, 14, 19]


def test_should_run_evaluation_still_runs_final_round_when_interval_is_large():
    scheduled = [
        epoch for epoch in range(3)
        if should_run_evaluation(epoch, total_epochs=3, eval_interval=10)
    ]
    assert scheduled == [2]


def test_default_eval_interval_is_ten():
    assert DEFAULT_EVAL_INTERVAL == 10
