import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codegraph_agent import CodeGraphAgent, RepositoryIndexer


ROOT = Path(__file__).resolve().parents[1]
MINI_REPO = ROOT / "examples" / "mini_repo"


class RepositoryIndexerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.trace_dir = Path(self.tmp.name) / "traces"

    def tearDown(self):
        self.tmp.cleanup()

    def test_indexer_builds_cross_file_graph(self):
        graph = RepositoryIndexer().build(MINI_REPO)
        names = {symbol.name for symbol in graph.symbols.values()}
        self.assertIn("calculate_total", names)
        self.assertIn("apply_tax", names)
        self.assertIn("CheckoutService", names)
        call_edges = [edge for edge in graph.index.edges if edge.kind == "calls"]
        self.assertTrue(any(edge.evidence == "apply_tax" for edge in call_edges))

    def test_agent_impact_analysis_mentions_callers(self):
        agent = CodeGraphAgent(MINI_REPO, trace_dir=self.trace_dir)
        response = agent.answer("If calculate_total changes, which code may be affected?")
        blob = str(response.evidence)
        self.assertIn("'name': 'calculate_total'", blob)
        self.assertIn("discount_total", blob)
        self.assertIn("services/checkout.py", blob)

    def test_agent_call_chain(self):
        agent = CodeGraphAgent(MINI_REPO, trace_dir=self.trace_dir)
        response = agent.answer("Show the call chain from quote_with_coupon")
        blob = response.answer + str(response.evidence)
        self.assertIn("discount_total", blob)
        self.assertIn("calculate_total", blob)


if __name__ == "__main__":
    unittest.main()
