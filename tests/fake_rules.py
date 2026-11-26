"""A rule-based stand-in for the LLM, used only by the tests and by tests/fake_ollama_server.py.

It lets every stage be tested without Ollama: the "model" answers from a few keyword rules,
so the tests check ARGUS's own logic (prompts, parsing, caching, scoring), not model quality.
"""

POSITIVE = ["support", "should not live in poverty", "helps", "barely affect employment", "modest increases"]
NEGATIVE = ["layoffs", "thin margins", "disagree", "ignores", "force"]


def last_field(prompt: str, name: str) -> str:
    """Value of the last line that starts with `name:` (the real case comes after any examples)."""
    value = ""
    for line in prompt.splitlines():
        if line.startswith(name + ":"):
            value = line[len(name) + 1:].strip()
    return value


def polarity(text: str) -> int:
    t = text.lower()
    if any(w in t for w in NEGATIVE):
        return -1
    if any(w in t for w in POSITIVE):
        return 1
    return 0


def fake_reply(prompt: str, keys: set) -> dict:
    """Answer for whichever stage the prompt belongs to, identified by the schema's field names."""
    statement = last_field(prompt, "Statement")
    if keys == {"stance", "explanation"} or keys == {"reasoning", "stance"}:
        p = polarity(statement)
        stance = "Favor" if p > 0 else "Against" if p < 0 else "Neutral"
        if "reasoning" in keys:
            return {"reasoning": f"fake reasoning about: {statement[:40]}", "stance": stance}
        return {"stance": stance, "explanation": f"fake explanation about: {statement[:40]}"}
    if keys == {"component", "identified_argument", "explanation"}:
        s = statement.lower()
        if s.startswith(("i support", "i disagree", "i still oppose", "i would go further")):
            component = "Claim"
        elif polarity(statement) != 0:
            component = "Premise"
        else:
            component = "None"
        return {"component": component,
                "identified_argument": "" if component == "None" else statement,
                "explanation": "fake explanation"}
    if keys == {"relation", "explanation"}:
        a, b = polarity(last_field(prompt, "Argument A")), polarity(last_field(prompt, "Argument B"))
        relation = "None" if a == 0 or b == 0 else "Support" if a == b else "Attack"
        return {"relation": relation, "explanation": "fake explanation"}
    raise ValueError(f"fake_reply does not know the schema with fields {sorted(keys)}")
