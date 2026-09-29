from localrag.config import load_config
from localrag.eval.metrics import mrr, ndcg_at_k, recall_at_k
from localrag.ingest.chunker import chunk_blocks, split_text
from localrag.retrieval.bm25 import BM25, tokenize
from localrag.retrieval.fusion import reciprocal_rank_fusion
from localrag.types import Block


def test_tokenize_keeps_section_numbers():
    assert tokenize("See 21 CFR 820.30(g) for design validation") == ["see", "21", "cfr", "820.30", "g", "design", "validation"]


def test_bm25_ranks_matching_doc_first(tmp_path):
    bm = BM25().fit(["a", "b", "c"], ["design validation of devices", "purchasing controls", "design review meetings"])
    assert bm.search("design validation")[0][0] == "a"
    bm.save(tmp_path / "bm25.json")
    assert BM25.load(tmp_path / "bm25.json").search("purchasing")[0][0] == "b"


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["x", "y", "z"], ["y", "x", "w"]], k=60)
    assert {fused[0][0], fused[1][0]} == {"x", "y"}
    assert fused[-1][0] in {"z", "w"}


def test_rrf_weights():
    fused = reciprocal_rank_fusion([["x"], ["y"]], weights=[2.0, 1.0])
    assert fused[0][0] == "x"


def test_split_text_respects_size_and_overlaps():
    text = " ".join(f"Sentence number {i} is here." for i in range(50))
    pieces = split_text(text, size=200, overlap=50)
    assert all(len(p) <= 200 for p in pieces)
    assert len(pieces) > 1
    # the last sentence of a piece reappears at the start of the next
    assert pieces[0].split(". ")[-1].rstrip(".") in pieces[1]


def test_chunker_builds_heading_path_and_stable_ids():
    blocks = [
        Block("heading", "Subpart C", "doc.pdf", 1, level=1),
        Block("heading", "820.30 Design controls", "doc.pdf", 1, level=2),
        Block("paragraph", "Each manufacturer shall establish procedures.", "doc.pdf", 1),
        Block("heading", "Subpart D", "doc.pdf", 2, level=1),
        Block("paragraph", "Document controls apply.", "doc.pdf", 2),
    ]
    chunks = chunk_blocks(blocks)
    assert [c.section for c in chunks] == ["Subpart C > 820.30 Design controls", "Subpart D"]
    assert chunks[0].id == chunk_blocks(blocks)[0].id


def test_chunks_never_cross_pages():
    blocks = [Block("paragraph", "Page one text here.", "d.pdf", 1),
              Block("paragraph", "Page two text here.", "d.pdf", 2)]
    assert [c.page for c in chunk_blocks(blocks)] == [1, 2]


def test_retrieval_metrics():
    gold = {("d", 3)}
    ranked = [("d", 1), ("d", 3), ("d", 3)]
    assert recall_at_k(ranked, gold, 1) == 0 and recall_at_k(ranked, gold, 2) == 1
    assert mrr(ranked, gold) == 0.5
    assert abs(ndcg_at_k(ranked, gold, 10) - 1 / 1.5849625) < 1e-6  # duplicate page not double counted


def test_config_overrides():
    cfg = load_config(None, ["retrieval.mode=dense", "agent.decompose=false", "retrieval.reranker_model=null"])
    assert cfg.retrieval.mode == "dense" and cfg.agent.decompose is False
    assert cfg.retrieval.reranker_model is None


def test_dedupe_merges_near_duplicate_claims():
    from localrag.agents import _dedupe
    from localrag.types import Claim
    out = _dedupe([Claim("Design verification confirms that design output meets design input requirements.", [4]),
                   Claim("Design verification shall confirm that design output meets design input requirements.", [2]),
                   Claim("Results are documented in the DHF.", [4])])
    assert len(out) == 2 and out[0].citations == [2, 4]


def test_refusal_claims_are_detected():
    from localrag.agents import _REFUSAL
    assert _REFUSAL.search("The EU MDR rules are not explicitly detailed in the provided passages.")
    assert not _REFUSAL.search("Design verification shall confirm that the design output meets the design input.")


def test_split_table_repeats_header():
    from localrag.ingest.chunker import split_table
    table = "| a | b |\n|---|---|\n" + "\n".join(f"| row{i} | {'x' * 20} |" for i in range(20))
    pieces = split_table(table, 200)
    assert len(pieces) > 1 and all(p.startswith("| a | b |\n|---|---|") for p in pieces)
    assert sum(p.count("row") for p in pieces) == 20


def test_parent_child_children_point_to_parent():
    from localrag.ingest.chunker import make_children
    blocks = [Block("heading", "Sec", "d.pdf", 1, level=1),
              Block("paragraph", " ".join(f"Sentence {i} is about design." for i in range(40)), "d.pdf", 1)]
    parents = chunk_blocks(blocks, size=2000)
    kids = make_children(parents, size=200, overlap=40)
    assert len(kids) > 1 and {k.parent_id for k in kids} == {parents[0].id}
    assert all(k.section == "Sec" for k in kids)


def test_splade_inverted_index_scoring():
    from localrag.retrieval.index import SpladeIndex
    idx = SpladeIndex(["a", "b"], {7: [(0, 2.0), (1, 0.5)], 9: [(1, 3.0)]})
    assert idx.score({7: 1.0, 9: 1.0}, k=5) == [("b", 3.5), ("a", 2.0)]
    assert idx.score({7: 1.0}, k=1) == [("a", 2.0)]


def test_splits_are_deterministic_and_by_page():
    from localrag.eval.dataset import assign_split, split_of
    assert assign_split("doc.pdf|3") == assign_split("doc.pdf|3")
    shares = [assign_split(f"k{i}") for i in range(3000)]
    assert 0.45 < shares.count("train") / 3000 < 0.55 and 0.30 < shares.count("test") / 3000 < 0.40
    assert split_of({"id": "x", "split": "dev"}) == "dev"


def test_bootstrap_ci_shrinks_with_n():
    from localrag.eval.runner import _ci
    small, large = _ci([0, 1] * 10, 500), _ci([0, 1] * 200, 500)
    assert small > large > 0


def _fake_nli(pairs):
    # "entails" iff every word of the claim appears in the passage
    def words(t):
        return set(t.lower().replace(".", "").split())
    return [1.0 if words(c) <= words(p) else 0.0 for c, p in pairs]


def test_answer_reward_components():
    from localrag.rl.rewards import AnswerExample, answer_reward
    ex = AnswerExample(passages=["design verification confirms outputs meet inputs.", "purchasing controls apply."],
                       gold=[1], answerable=True)
    good = '{"claims": [{"text": "verification confirms outputs meet inputs", "citations": [1]}]}'
    wrong_cite = '{"claims": [{"text": "verification confirms outputs meet inputs", "citations": [2]}]}'
    r_good, parts = answer_reward(good, ex, _fake_nli)
    assert parts["support"] == 1.0 and parts["gold"] == 1.0 and r_good == 1.5
    assert answer_reward(wrong_cite, ex, _fake_nli)[0] == 0.0          # unsupported and not gold
    assert answer_reward('{"claims": [{"text": "x", "citations": [7]}]}', ex, _fake_nli)[0] == -0.5
    assert answer_reward("not json", ex, _fake_nli)[0] == -1.0
    assert answer_reward('{"claims": []}', ex, _fake_nli)[0] == -1.0    # refusing an answerable question


def test_answer_reward_rewards_refusing_unanswerable():
    from localrag.rl.rewards import AnswerExample, answer_reward
    ex = AnswerExample(passages=["purchasing controls apply."], gold=[], answerable=False)
    assert answer_reward('{"claims": []}', ex, _fake_nli)[0] == 1.0
    hallucinated = '{"claims": [{"text": "EU MDR rule 11 applies", "citations": [1]}]}'
    assert answer_reward(hallucinated, ex, _fake_nli)[0] < 0


def test_decompose_reward_is_gold_recall():
    from types import SimpleNamespace as NS
    from localrag.rl.rewards import decompose_reward
    def retrieve(q, subs, k):
        return [NS(chunk=NS(source="a", page=1))] + ([NS(chunk=NS(source="b", page=2))] if "b" in " ".join(subs) else [])
    gold = {("a", 1), ("b", 2)}
    assert decompose_reward('{"queries": ["about b"]}', "q", gold, retrieve)[0] == 1.0
    assert decompose_reward('{"queries": ["other"]}', "q", gold, retrieve)[0] == 0.5
    assert decompose_reward("oops", "q", gold, retrieve)[0] == -1.0


def test_coverage_rewards_complete_answers():
    from localrag.rl.rewards import AnswerExample, answer_reward
    passages = ["design verification confirms outputs meet inputs. results are documented in the dhf."]
    ref = "Verification confirms outputs meet inputs. Results are documented in the DHF."
    ex = AnswerExample(passages, gold=[1], answerable=True, reference=ref)
    partial = '{"claims": [{"text": "verification confirms outputs meet inputs", "citations": [1]}]}'
    full = ('{"claims": [{"text": "verification confirms outputs meet inputs", "citations": [1]}, '
            '{"text": "results are documented in the dhf", "citations": [1]}]}')
    r_partial, p1 = answer_reward(partial, ex, _fake_nli)
    r_full, p2 = answer_reward(full, ex, _fake_nli)
    assert p1["coverage"] == 0.5 and p2["coverage"] == 1.0 and r_full > r_partial
