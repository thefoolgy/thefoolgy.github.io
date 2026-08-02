#!/usr/bin/env python3
"""Build the browser dataset for the 41 persistent HotpotQA failures."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


CLASS_LABELS = {
    "boundary": "答案边界 / 规范化",
    "content": "内容或关系判断错误",
    "data": "数据 / 证据歧义",
    "mixed": "条件间混合错误",
}

MODEL_LABELS = {
    "baseline": "Baseline",
    "paragraph_oracle": "Paragraph Oracle",
    "sentence_oracle": "Sentence Oracle",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_prediction(text: str) -> str:
    if text == "[NO_FINAL_ANSWER]":
        return "未生成可解析的最终答案"
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    raw_items = load_json(args.raw)
    translations = {
        item["question_id"]: item for item in load_json(args.translations)
    }
    review = load_json(args.review)

    if len(raw_items) != 41:
        raise ValueError(f"Expected 41 raw cases, got {len(raw_items)}")
    qids = [item["question_id"] for item in raw_items]
    if len(set(qids)) != 41:
        raise ValueError("Question IDs are not unique")
    if set(qids) != set(translations) or set(qids) != set(review):
        raise ValueError("Raw, translation, and manual-review IDs do not match")

    cases = []
    for source in raw_items:
        qid = source["question_id"]
        translated = translations[qid]
        checked = review[qid]
        review_class = checked["review_class"]
        if review_class not in CLASS_LABELS:
            raise ValueError(f"Unknown review class {review_class} for {qid}")

        translated_evidence = checked.get("evidence_zh", translated["evidence_zh"])
        if len(translated_evidence) != len(source["evidence"]):
            raise ValueError(f"Evidence group count changed for {qid}")

        evidence = []
        for original, chinese in zip(
            source["evidence"], translated_evidence, strict=True
        ):
            original_facts = [sentence["text"] for sentence in original["sentences"]]
            if len(original_facts) != len(chinese["facts_zh"]):
                raise ValueError(f"Evidence sentence count changed for {qid}")
            evidence.append(
                {
                    "title": original["title"],
                    "title_zh": chinese["title_zh"],
                    "facts": original_facts,
                    "facts_zh": chinese["facts_zh"],
                }
            )

        models = {}
        translated_predictions = checked.get(
            "predictions_zh", translated["predictions_zh"]
        )
        if set(translated_predictions) != set(MODEL_LABELS):
            raise ValueError(f"Prediction translation keys changed for {qid}")
        for model_name, label in MODEL_LABELS.items():
            model = source["models"][model_name]
            models[model_name] = {
                "label": label,
                "prediction": normalize_prediction(model["prediction"]),
                "prediction_zh": translated_predictions[model_name],
                "token_f1": model["token_f1"],
                "format_valid": model["format_valid"],
                "num_steps": model["num_steps"],
                "generated_queries": model["generated_queries"],
            }

        cases.append(
            {
                "index": source["index"],
                "question_id": qid,
                "source_index": source["source_index"],
                "review_class": review_class,
                "review_class_zh": CLASS_LABELS[review_class],
                "subclass": checked["subclass"],
                "question": source["question"],
                "question_zh": checked.get("question_zh", translated["question_zh"]),
                "ground_truth": source["ground_truth"],
                "ground_truth_zh": checked.get(
                    "ground_truth_zh", translated["ground_truth_zh"]
                ),
                "diagnosis_zh": checked["diagnosis_zh"],
                "evidence": evidence,
                "models": models,
            }
        )

    class_counts = Counter(item["review_class"] for item in cases)
    expected_counts = {"boundary": 25, "content": 6, "data": 9, "mixed": 1}
    if dict(class_counts) != expected_counts:
        raise ValueError(
            f"Unexpected review counts: {dict(class_counts)} != {expected_counts}"
        )

    dataset = {
        "version": "1.0.0",
        "generated_at": "2026-08-02",
        "experiment": "HotpotQA persistent Bridge failures under three conditions",
        "state": "000",
        "state_legend": "Baseline EM, Paragraph Oracle EM, Sentence Oracle EM",
        "n": len(cases),
        "class_labels": CLASS_LABELS,
        "class_counts": expected_counts,
        "method": "Qwen3-4B-Instruct-2507 translation draft followed by manual question-level review; English originals are retained.",
        "source_sha256": {
            "raw": sha256(args.raw),
            "translations": sha256(args.translations),
            "manual_review": sha256(args.review),
        },
        "cases": cases,
    }

    args.output_json.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "n": len(cases),
                "class_counts": expected_counts,
                "output_json": str(args.output_json),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
