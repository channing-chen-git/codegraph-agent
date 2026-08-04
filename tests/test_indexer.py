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
        agent = CodeGraphAgent(MINI_REPO, trace_dir=self.trace_dir, use_llm=False, require_llm=False)
        response = agent.answer("If calculate_total changes, which code may be affected?")
        blob = str(response.evidence)
        self.assertIn("'name': 'calculate_total'", blob)
        self.assertIn("discount_total", blob)
        self.assertIn("services/checkout.py", blob)

    def test_agent_call_chain(self):
        agent = CodeGraphAgent(MINI_REPO, trace_dir=self.trace_dir, use_llm=False, require_llm=False)
        response = agent.answer("Show the call chain from quote_with_coupon")
        blob = response.answer + str(response.evidence)
        self.assertIn("discount_total", blob)
        self.assertIn("calculate_total", blob)

    def test_function_calling_schemas_are_exposed(self):
        agent = CodeGraphAgent(MINI_REPO, trace_dir=self.trace_dir, use_llm=False, require_llm=False)
        schemas = agent.tools.function_schemas()
        names = {schema["function"]["name"] for schema in schemas}
        self.assertIn("impact_analysis", names)
        self.assertIn("test_recommendations", names)
        for schema in schemas:
            self.assertEqual(schema["type"], "function")
            self.assertIn("parameters", schema["function"])

    def test_agent_can_reuse_prebuilt_graph(self):
        graph = RepositoryIndexer().build(MINI_REPO)
        agent = CodeGraphAgent(
            MINI_REPO,
            trace_dir=self.trace_dir,
            use_llm=False,
            require_llm=False,
            graph=graph,
        )
        self.assertIs(agent.graph, graph)
        response = agent.answer("If calculate_total changes, which code may be affected?")
        self.assertIn("impact_analysis", response.tools_used)

    def test_pr_change_analysis_uses_diff_evidence(self):
        agent = CodeGraphAgent(MINI_REPO, trace_dir=self.trace_dir, use_llm=False, require_llm=False)
        response = agent.answer("Analyze this PR diff and recommend tests")
        self.assertIn("pr_change_analysis", response.tools_used)
        blob = str(response.evidence)
        self.assertIn("changes.diff", blob)
        self.assertIn("calculate_total", blob)
        self.assertIn("impact_by_changed_symbol", blob)

    def test_coverage_gap_analysis_uses_coverage_evidence(self):
        agent = CodeGraphAgent(MINI_REPO, trace_dir=self.trace_dir, use_llm=False, require_llm=False)
        response = agent.answer("Find coverage gaps for calculate_total")
        self.assertIn("coverage_gap_analysis", response.tools_used)
        blob = str(response.evidence)
        self.assertIn("coverage.json", blob)
        self.assertIn("file_coverage_percent", blob)
        self.assertIn("missing_lines_in_symbol", blob)

    def test_runtime_trace_analysis_supplements_static_graph(self):
        agent = CodeGraphAgent(MINI_REPO, trace_dir=self.trace_dir, use_llm=False, require_llm=False)
        response = agent.answer("Use runtime trace evidence for calculate_total")
        self.assertIn("runtime_trace_analysis", response.tools_used)
        self.assertIn("impact_analysis", response.tools_used)
        blob = str(response.evidence)
        self.assertIn("runtime_traces.json", blob)
        self.assertIn("POST /checkout/quote", blob)


if __name__ == "__main__":
    unittest.main()
