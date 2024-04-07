# Nothing here

from src.const import all_labels


class MetricForPII:
    def __init__(self, metric):
        self.metric = metric

    def __call__(self, p):
        predictions, labels = p
        predictions = predictions.argmax(axis=2)
        # Remove ignored index (special tokens)
        true_predictions = [
            [all_labels[p] for (p, l) in zip(prediction, label) if l != 0 and p != 0]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [all_labels[l] for (p, l) in zip(prediction, label) if l != 0 and p != 0]
            for prediction, label in zip(predictions, labels)
        ]
        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }