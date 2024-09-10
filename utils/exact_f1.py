from collections import Counter
from typing import Tuple, Dict, List, Set


class ExactF1:

    def __init__(self) -> None:
        super(ExactF1, self).__init__()

        self._true_positives: Dict[str, int] = Counter()
        self._false_positives: Dict[str, int] = Counter()
        self._false_negatives: Dict[str, int] = Counter()

    def reset(self) -> None:
        self._true_positives.clear()
        self._false_positives.clear()
        self._false_negatives.clear()

    def __call__(self, predictions: List[List[str]], golds: List[List[str]]) -> None:
        assert len(predictions) == len(golds), f"batch error:\n{predictions}\n{golds}"
        for prediction, gold in zip(predictions, golds):
            assert len(prediction) == len(gold), f"pair error:\n{prediction}\n{gold}"
            pred_spans = self.__get_spans(prediction)
            gold_spans = self.__get_spans(gold)

            for span in pred_spans:
                if span in gold_spans:
                    self._true_positives[span[2]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[2]] += 1
            for span in gold_spans:
                self._false_negatives[span[2]] += 1

    @staticmethod
    def __get_spans(tag_sequence: List[str]) -> Set[Tuple[int, int, str]]:
        spans = set()
        span_start, span_end, span_tag = -1, -1, None
        for i, full_tag in enumerate(tag_sequence):
            bio_tag, type_tag = full_tag[0], full_tag[2:]
            if bio_tag == "B":
                if span_tag is not None:
                    spans.add((span_start, span_end, span_tag))
                span_start, span_end, span_tag = i, i + 1, type_tag
            elif bio_tag == "I":
                if type_tag == span_tag:
                    span_end += 1
                else:
                    if span_tag is not None:
                        spans.add((span_start, span_end, span_tag))
                    span_start, span_end, span_tag = -1, -1, None
            elif bio_tag == "O":
                if span_tag is not None:
                    spans.add((span_start, span_end, span_tag))
                span_start, span_end, span_tag = -1, -1, None
            else:
                raise ValueError(f"Invalid BIO tag: {bio_tag}")
        if span_tag is not None:
            spans.add((span_start, span_end, span_tag))
        return spans

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        all_tags = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}

        for tag in all_tags:
            precision, recall, f1_measure = self.__compute_metrics(
                self._true_positives[tag], self._false_positives[tag], self._false_negatives[tag]
            )
            all_metrics[f"P-{tag}"] = precision
            all_metrics[f"R-{tag}"] = recall
            all_metrics[f"F1-{tag}"] = f1_measure

        precision, recall, f1_measure = self.__compute_metrics(
            sum(self._true_positives.values()),
            sum(self._false_positives.values()),
            sum(self._false_negatives.values()),
        )
        all_metrics["P"] = precision
        all_metrics["R"] = recall
        all_metrics["F1"] = f1_measure

        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def __compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0.
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.
        f1_measure = 2.0 * (precision * recall) / (precision + recall) if precision + recall > 0. else 0.
        return precision*100, recall*100, f1_measure*100
