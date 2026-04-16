import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flpoison.fl.eval_schedule import DEFAULT_EVALUATE, should_run_evaluation


def test_should_run_evaluation_when_enabled():
    assert should_run_evaluation(True) is True


def test_should_not_run_evaluation_when_disabled():
    assert should_run_evaluation(False) is False


def test_default_evaluate_is_disabled():
    assert DEFAULT_EVALUATE is False
