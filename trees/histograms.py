
"""
Histograms allow for chunking data so that large data arrays can be processed much more quickly.

"""
from dataclasses import dataclass, field
from math import sqrt, floor, ceil, log2, pow
from typing import Callable, List
from statistics import stdev, median


@dataclass
class HistogramAlgo:
    """
    Various algorithms differ in how to calculate the number of bins and how wide the bin is.
    """
    bins: Callable[[int, List[float]], int]
    bin_width: Callable[[int, List[float]], float]

@dataclass
class HistogramBin:
    min_value: int
    max_value: int
    values: List[float] = field(default_factory=list)

    def add(self, value: float) -> bool:
        if self.min_value <= value < self.max_value:
            self.values.append(value)
            return True
        return False

histogram_algorithms = {
    'Square': HistogramAlgo(lambda values: floor(sqrt(len(values))),
                            lambda values: (max(values)-min(values))/sqrt(len(values))),
    'Sturges': HistogramAlgo(lambda values: ceil(log2(len(values)))+1,
                             lambda values: (max(values) - min(values)) / (ceil(log2(len(values)))+1)),
    'Rice': HistogramAlgo(lambda values: floor(2*pow(len(values), 1/3.)),
                          lambda values: (max(values) - min(values)) / floor(2*pow(len(values), 1/3.))),
    'Scott': HistogramAlgo(lambda values: (max(values)-min(values))/(3.5*stdev(values)/pow(len(values), 1/3.)),
                           lambda values: 3.5*stdev(values)/pow(len(values), 1/3.)),
    'Freedman-Diaconis': HistogramAlgo(lambda n, values: (max(values)-min(values))/(2*median(values)/pow(n,1/3.)),
                                       lambda n, values: 2*median(values)/pow(n,1/3.))
}


def make_histogram_bins(values: List[float], algorithm:str = 'Square'):
    """
    Applies a given histogram algorithm to make bins of the list of data.
    """
    algo = histogram_algorithms[algorithm]
    num_bins = algo.bins(values)
    bins_width = algo.bins_width(values)
    bin_ranges = []
    start_val = min(values)
    for i in range(len(num_bins)):
        bin_ranges.append((start_val, start_val+bins_width))
    bins = [HistogramBin(min_value, max_value) for (min_value, max_value) in bin_ranges]

    for val in values:
        found = False
        for bin in bins:
            if bin.add(val):
                found = True
                break
        if not found:
            print(f"Couldn't find histogram bucket: {val} in buckets: {bins}")

    return bins
