"""LLM-as-judge for answer quality, using a larger local model than the one under test."""
from __future__ import annotations

from localrag.llm import LLM

CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {"reasoning": {"type": "string", "maxLength": 400},
                   "verdict": {"type": "string", "enum": ["correct", "partially_correct", "incorrect"]}},
    "required": ["reasoning", "verdict"],
}
VERDICT_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


def correctness(judge: LLM, question: str, reference: str, answer: str) -> float:
    prompt = f"""Grade a system answer against a reference answer.
Question: {question}
Reference answer: {reference}
System answer: {answer}

"correct": contains the key facts of the reference and nothing contradicting it (extra correct detail is fine).
"partially_correct": some key facts are present, others missing.
"incorrect": wrong, contradicting, off-topic, or a refusal."""
    out = judge.json([{"role": "user", "content": prompt}], CORRECTNESS_SCHEMA, max_tokens=300, temperature=0)
    return VERDICT_SCORE[out["verdict"]]


def faithfulness(judge: LLM, claims: list[dict]) -> list[bool]:
    """For each claim ({"text", "citations", "passages"}), is it supported by the passages it cites?
    Uncited claims are unsupported by definition."""
    cited = [c for c in claims if c["citations"]]
    if not cited:
        return [False] * len(claims)
    listing = []
    for i, c in enumerate(cited, start=1):
        passages = "\n".join(f"  - {p}" for p in c["passages"])
        listing.append(f"Claim {i}: {c['text']}\nCited passages:\n{passages}")
    schema = {"type": "object",
              "properties": {"supported": {"type": "array", "minItems": len(cited), "maxItems": len(cited),
                                           "items": {"type": "boolean"}}},
              "required": ["supported"]}
    prompt = ("For each claim, answer true if the cited passages state or directly imply the claim, "
              "false otherwise.\n\n" + "\n\n".join(listing))
    out = judge.json([{"role": "user", "content": prompt}], schema, max_tokens=300, temperature=0)
    verdicts = iter(out["supported"])
    return [next(verdicts) if c["citations"] else False for c in claims]
