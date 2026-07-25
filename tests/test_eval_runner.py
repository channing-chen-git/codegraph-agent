import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codegraph_agent.eval_runner import run_eval


ROOT = Path(__file__).resolve().parents[1]


class EvalRunnerTest(unittest.TestCase):
    def test_eval_runner_smoke(self):
        with TemporaryDirectory() as tmp:
            result = run_eval(
                ROOT / "examples" / "mini_repo",
                ROOT / "eval" / "tasks.json",
                trace_dir=Path(tmp) / "eval_traces",
            )
        self.assertEqual(result["total"], 4)
        self.assertGreaterEqual(result["passed"], 3)


if __name__ == "__main__":
    unittest.main()
