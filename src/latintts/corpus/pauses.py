from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from latintts.corpus.domain import CorpusFailure
from latintts.corpus.vad import SpeechInterval

PauseKind = Literal["short", "long"]


@dataclass(frozen=True, slots=True)
class PauseInterval:
    start_sample: int
    end_sample: int
    duration_seconds: float
    kind: PauseKind


@dataclass(frozen=True, slots=True)
class PauseAnalysis:
    pauses: tuple[PauseInterval, ...]
    threshold_seconds: float
    short_median_seconds: float
    long_median_seconds: float


def _validate_parameters(
    sample_rate: int,
    minimum_gap_ms: int,
    minimum_gap_count: int,
    separation_ratio: float,
    maximum_iterations: int,
    minimum_cluster_size: int,
    maximum_cluster_imbalance_ratio: float,
) -> None:
    for value, name, lower_bound in (
        (sample_rate, "sample_rate", 1),
        (minimum_gap_ms, "minimum_gap_ms", 0),
        (minimum_gap_count, "minimum_gap_count", 1),
        (maximum_iterations, "maximum_iterations", 1),
        (minimum_cluster_size, "minimum_cluster_size", 2),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
        if value < lower_bound:
            raise ValueError(f"{name} is out of range")
    if type(separation_ratio) not in (int, float):
        raise TypeError("separation_ratio must be a number")
    if not math.isfinite(separation_ratio) or separation_ratio <= 1:
        raise ValueError("separation_ratio must be finite and greater than one")
    if type(maximum_cluster_imbalance_ratio) not in (int, float):
        raise TypeError("maximum_cluster_imbalance_ratio must be a number")
    if not math.isfinite(maximum_cluster_imbalance_ratio) or maximum_cluster_imbalance_ratio < 1:
        raise ValueError("maximum_cluster_imbalance_ratio must be finite and at least one")
    if minimum_gap_count < minimum_cluster_size * 2:
        raise ValueError("minimum_gap_count must allow two minimum-size clusters")


def classify_pauses(
    speech: tuple[SpeechInterval, ...],
    *,
    sample_rate: int,
    minimum_gap_ms: int,
    minimum_gap_count: int,
    separation_ratio: float,
    maximum_iterations: int,
    minimum_cluster_size: int,
    maximum_cluster_imbalance_ratio: float,
) -> PauseAnalysis:
    _validate_parameters(
        sample_rate,
        minimum_gap_ms,
        minimum_gap_count,
        separation_ratio,
        maximum_iterations,
        minimum_cluster_size,
        maximum_cluster_imbalance_ratio,
    )
    if type(speech) is not tuple or any(
        type(interval) is not SpeechInterval for interval in speech
    ):
        raise TypeError("speech must be a tuple of SpeechInterval values")
    if any(left.end_sample > right.start_sample for left, right in pairwise(speech)):
        raise ValueError("speech intervals must be ordered and non-overlapping")
    gaps = [
        (left.end_sample, right.start_sample)
        for left, right in pairwise(speech)
        if (right.start_sample - left.end_sample) * 1000 >= minimum_gap_ms * sample_rate
    ]
    if len(gaps) < minimum_gap_count:
        raise CorpusFailure("PAUSE_CLASSES_AMBIGUOUS", "too few eligible internal pauses")

    durations = [(end - start) / sample_rate for start, end in gaps]
    values = [math.log(value) for value in durations]
    ordered = sorted(values)
    centers = [ordered[len(ordered) // 4], ordered[(3 * len(ordered)) // 4]]
    labels: list[int] | None = None
    for _ in range(maximum_iterations):
        next_labels = [
            min(range(2), key=lambda cluster: (abs(value - centers[cluster]), cluster))
            for value in values
        ]
        groups = [
            [value for value, label in zip(values, next_labels, strict=True) if label == cluster]
            for cluster in range(2)
        ]
        if any(not group for group in groups):
            raise CorpusFailure("PAUSE_CLASSES_AMBIGUOUS", "pause cluster is empty")
        next_centers = [statistics.fmean(group) for group in groups]
        if next_labels == labels and next_centers == centers:
            break
        labels, centers = next_labels, next_centers
    assert labels is not None
    cluster_sizes = [labels.count(cluster) for cluster in range(2)]
    if min(cluster_sizes) < minimum_cluster_size:
        raise CorpusFailure("PAUSE_CLASSES_AMBIGUOUS", "pause cluster size is unreliable")
    if max(cluster_sizes) / min(cluster_sizes) > maximum_cluster_imbalance_ratio:
        raise CorpusFailure("PAUSE_CLASSES_AMBIGUOUS", "pause clusters are imbalanced")

    medians = [
        statistics.median(
            duration for duration, label in zip(durations, labels, strict=True) if label == cluster
        )
        for cluster in range(2)
    ]
    short_label = 0 if medians[0] < medians[1] else 1
    short_median = medians[short_label]
    long_median = medians[1 - short_label]
    if long_median / short_median < separation_ratio:
        raise CorpusFailure("PAUSE_CLASSES_AMBIGUOUS", "pause clusters are not separated")
    threshold = math.sqrt(short_median * long_median)
    pauses = tuple(
        PauseInterval(start, end, duration, "short" if duration < threshold else "long")
        for (start, end), duration in zip(gaps, durations, strict=True)
    )
    return PauseAnalysis(pauses, threshold, short_median, long_median)
