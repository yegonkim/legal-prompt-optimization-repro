"""Pinned public-source formatting, extracted without dataset-loading side effects."""
import ast,hashlib,re

def _text(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected nonempty text, got {value!r}")
    return re.sub(r"<br\s*/?>", "\n", value).strip()

def _mcq(question: str, choices: list[str], answer: str, labels: list[str]) -> dict:
    if len(choices) != len(labels) or answer not in labels:
        raise ValueError("Invalid multiple-choice labels or answer")
    options = "\n".join(f"{label}. {_text(choice)}" for label, choice in zip(labels, choices))
    return {
        "question": f"{_text(question)}\n\n{options}\n\nReturn only the correct option label as your final answer.",
        "answer": answer,
    }

def format_lexam(row: dict) -> dict:
    choices = ast.literal_eval(row["choices"]) if isinstance(row["choices"], str) else row["choices"]
    if not isinstance(choices, list) or len(choices) not in (4, 8, 16, 32):
        raise ValueError("LEXam choices must be a list of 4, 8, 16, or 32 strings")
    gold = row["gold"]
    if not isinstance(gold, int) or not 0 <= gold < len(choices):
        raise ValueError(f"Invalid LEXam gold index: {gold!r}")
    # Numeric labels also cover the 32-choice subset without ambiguous letters.
    labels = [str(i + 1) for i in range(len(choices))]
    question = (
        f"Course: {_text(row['course'])}\nJurisdiction: {_text(row['jurisdiction'])}\n\n"
        f"{_text(row['question'])}"
    )
    return _mcq(question, choices, labels[gold], labels)

def format_lar_echr(row: dict, full_facts: bool = False) -> dict:
    facts = row["facts"]
    if full_facts:
        paragraphs = ast.literal_eval(row["complete_facts"]) if isinstance(row["complete_facts"], str) else row["complete_facts"]
        if not isinstance(paragraphs, list) or not paragraphs:
            raise ValueError("LAR-ECHR complete_facts must be a nonempty list")
        facts = "\n\n".join(_text(paragraph) for paragraph in paragraphs)
    question = (
        f"Facts:\n{_text(facts)}\n\nPrevious arguments:\n{_text(row['context'])}\n\n"
        "Which argument correctly continues this chain of legal arguments?"
    )
    return _mcq(question, [row[key] for key in "abcd"], row["label"], list("ABCD"))

def format_barexamqa(row: dict, evidence: bool = True) -> dict:
    required = ["question", "choice_a", "choice_b", "choice_c", "choice_d", "answer"]
    if evidence:
        required.append("gold_passage")
    missing = [key for key in required if not isinstance(row[key], str) or not row[key].strip()]
    if missing:
        raise ValueError(f"BarExamQA row {row.get('idx', '<unknown>')} has empty required fields: {', '.join(missing)}")
    # Some source questions are self-contained and have no separate fact pattern.
    question = f"Facts:\n{_text(row['prompt'])}\n\n" if row["prompt"] else ""
    if evidence:
        question += f"Supporting passage:\n{_text(row['gold_passage'])}\n\n"
    question += f"Question: {_text(row['question'])}"
    return _mcq(question, [row[f"choice_{key}"] for key in "abcd"], row["answer"], list("ABCD"))

def _choice_index(value, count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)) or not re.fullmatch(r"\d+", str(value)):
        raise ValueError("Expected a zero-based integer answer index")
    index = int(value)
    if not 0 <= index < count:
        raise ValueError("Answer index outside available options")
    return index

def format_casehold(row: dict) -> dict:
    context = _text(row["citing_prompt"])
    choices = [_text(row[f"holding_{i}"]) for i in range(5)]
    index = _choice_index(row["label"], 5)
    formatted = _mcq(f"Citing context:\n{context}\n\nWhich holding best completes the cited legal context?",
                     choices, list("ABCDE")[index], list("ABCDE"))
    formatted.update(choice_count=5, record_id=f"casehold:all:{row['source_split']}:{row['example_id']}",
                     group_id=hashlib.sha256(re.sub(r"\s+", " ", context).encode()).hexdigest())
    return formatted

