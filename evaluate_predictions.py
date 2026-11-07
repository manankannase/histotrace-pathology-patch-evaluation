"""Evaluate exported binary patch predictions using the Python standard library.

The CSV must contain id,label,probability. Labels are 0 or 1 and probability is
the model's probability of label 1. Undefined metrics are written as JSON null.
"""

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path


def load_predictions(path):
    predictions = []
    identifiers = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        fields = reader.fieldnames
        if not fields or len(set(fields)) != len(fields):
            raise ValueError("CSV headers must be present and unique")
        if not {"id", "label", "probability"}.issubset(fields):
            raise ValueError("CSV must include id,label,probability columns")

        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Line {reader.line_num}: inconsistent column count")
            identifier = row["id"].strip()
            if not identifier:
                raise ValueError(f"Line {reader.line_num}: empty patch ID")
            if identifier in identifiers:
                raise ValueError(f"Line {reader.line_num}: duplicate patch ID")
            if row["label"].strip() not in {"0", "1"}:
                raise ValueError(f"Line {reader.line_num}: label must be 0 or 1")
            try:
                probability = float(row["probability"])
            except ValueError:
                raise ValueError(
                    f"Line {reader.line_num}: probability must be numeric"
                ) from None
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(
                    f"Line {reader.line_num}: probability must be finite and in [0, 1]"
                )
            identifiers.add(identifier)
            predictions.append((int(row["label"]), probability))

    if not predictions:
        raise ValueError("CSV contains no predictions")
    return predictions


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def evaluate(predictions, threshold):
    true_positive = false_positive = true_negative = false_negative = 0
    squared_error = 0.0
    for label, probability in predictions:
        predicted = probability >= threshold
        true_positive += int(label == 1 and predicted)
        false_positive += int(label == 0 and predicted)
        true_negative += int(label == 0 and not predicted)
        false_negative += int(label == 1 and not predicted)
        squared_error += (probability - label) ** 2

    positives = true_positive + false_negative
    negatives = true_negative + false_positive
    seen = positive_seen = 0
    ordered_pairs = average_precision = 0.0
    ranked = sorted(predictions, key=lambda item: item[1], reverse=True)
    for _, group in itertools.groupby(ranked, key=lambda item: item[1]):
        labels = [label for label, _ in group]
        group_positive = sum(labels)
        group_negative = len(labels) - group_positive
        # A positive above a negative wins. Equal scores count as half a win.
        ordered_pairs += group_negative * (positive_seen + 0.5 * group_positive)
        positive_seen += group_positive
        seen += len(labels)
        if positives:
            average_precision += (group_positive / positives) * (positive_seen / seen)

    return {
        "samples": len(predictions),
        "positive_samples": positives,
        "negative_samples": negatives,
        "positive_label": 1,
        "threshold": threshold,
        "prediction_rule": "probability >= threshold",
        "confusion_matrix": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
        "accuracy": (true_positive + true_negative) / len(predictions),
        "precision": ratio(true_positive, true_positive + false_positive),
        "sensitivity_recall": ratio(true_positive, positives),
        "specificity": ratio(true_negative, negatives),
        "f1": ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative),
        "roc_auc": ratio(ordered_pairs, positives * negatives),
        "average_precision": average_precision if positives else None,
        "brier_score": squared_error / len(predictions),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="CSV of held-out predictions")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, help="Write JSON to a new file")
    arguments = parser.parse_args()
    if not math.isfinite(arguments.threshold) or not 0 <= arguments.threshold <= 1:
        parser.error("--threshold must be finite and in [0, 1]")
    try:
        report = evaluate(load_predictions(arguments.predictions), arguments.threshold)
        rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if arguments.output:
            with arguments.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
        else:
            sys.stdout.write(rendered)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(2, f"Evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
