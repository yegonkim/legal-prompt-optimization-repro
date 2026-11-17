"""Exercise configured generation features using synthetic problems only."""

import copy
from pathlib import Path

from .io import public, write


def smoke(config, out, backend_factory, include_proposer=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "synthetic_only": True, "checks": {}}
    write(out / "resolved-config.json", config)

    def require(records, name):
        valid = sum(bool(r.get("valid")) for r in records)
        report["checks"][name] = {"draws": len(records), "valid": valid}
        if valid != len(records):
            raise RuntimeError(
                f"{name}: {valid}/{len(records)} valid outputs; inspect {out}/cache"
            )

    try:
        backend = backend_factory(config, out / "cache")
        report["endpoints"] = backend.preflight(include_proposer=include_proposer)
        kind = {"mae": "regression", "brier": "probability"}.get(
            config["task"]["metric"], "classification"
        )
        row = {
            "record_id": "smoke-synthetic",
            "task_type": kind,
            "question": "The rule says a signed contract is valid. This contract is signed. Is it valid? Choose Yes or No.",
            "public_labels": ["Yes", "No"],
        }
        if kind != "classification":
            row.pop("public_labels")
            row["question"] = (
                "A payment of 2 euros plus 3 euros totals how many euros?"
                if kind == "regression"
                else "A fair coin is tossed once. What is the probability of heads?"
            )
        response = config["prompts"]["response"]
        judge = config["prompts"]["judge"]
        draws = backend.generate([row] * 4, response, [42, 43, 44, 45])
        require(draws, "response")
        if config["protocol"] == "legal_gepa_v1":
            from ..core import frame

            judge_row = public(
                frame(row, 0, [dict(r, draw=i) for i, r in enumerate(draws)])
            )
            judge = response + "\n\n" + judge
        else:
            judge_row = {**row, "candidates": draws}
        require(
            backend.generate([judge_row], judge, [46], judge=True),
            "four_response_judge",
        )
        if config["generation"]["answer_forcing"]["enabled"]:
            # Force an actual length finish; keep all sampling settings unchanged.
            capped = copy.deepcopy(config)
            capped["generation"]["response"]["max_tokens"] = 1
            capped["generation"]["judge"]["max_tokens"] = 1
            forcing = backend_factory(capped, out / "forcing-cache")
            write(out / "forcing-config.json", capped)
            for task, question in [
                ("classification", "Choose Yes: is a signed contract signed?"),
                ("regression", "What is 2 plus 3? Give a numeric answer."),
            ]:
                probe = {
                    "record_id": "smoke-forcing-" + task,
                    "task_type": task,
                    "question": question,
                }
                if task == "classification":
                    probe["public_labels"] = ["Yes", "No"]
                for is_judge in [False, True]:
                    data = {**probe, "candidates": draws} if is_judge else probe
                    result = forcing.generate(
                        [data], judge if is_judge else response, [47], judge=is_judge
                    )
                    name = f"forcing_{task}_{'judge' if is_judge else 'response'}"
                    require(result, name)
                    if not result[0].get("answer_forcing", {}).get("recovered"):
                        raise RuntimeError(
                            f"{name}: answer-forcing path was not exercised successfully"
                        )
                    report["checks"][name]["recovered"] = True
        if include_proposer:
            text = backend.reflect(
                "Improve this instruction: 'Solve the problem.' For a synthetic arithmetic task, "
                "the model answered 4 to 2+3; the correct answer is 5. Return one short reusable "
                "instruction inside triple backticks, preserving JSON reasoning and answer fields.",
                call_number=1,
            )
            report["checks"]["reflection"] = {"nonempty_final_text": bool(text.strip())}
            if not text.strip():
                raise RuntimeError("Reflection returned no final text")
        report["status"] = "passed"
        return report
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write(out / "smoke.json", report)
