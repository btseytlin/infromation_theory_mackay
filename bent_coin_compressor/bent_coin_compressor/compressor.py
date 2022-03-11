import math
import numpy as np
import scipy.stats
from numpy.typing import NDArray
from typing import Dict, Hashable, List
from itertools import combinations, product


def p_binomial(n: int, k: int, p: float) -> float:
    return math.factorial(n)/(math.factorial(k)*math.factorial(n-k)) * p**k * (1-p)**(n-k)


def gen_truth_table(n: int) -> NDArray:
    for n_ones in range(0, n+1):
        for ones_pos in combinations(range(n), n_ones):
            bstring = np.zeros(n).astype(int)
            bstring[list(ones_pos)] = 1
            yield bstring


def get_subset_cutoff(n_bits: int, one_proba: float, error_proba: float) -> int:
    total_proba = 0
    i = 0
    for n_ones in range(0, n_bits+1):
        if total_proba >= (1-error_proba):
            return i

        n_combs = math.factorial(n_bits)/(math.factorial(n_ones)*math.factorial(n_bits-n_ones))
        proba = p_binomial(n_bits, n_ones, one_proba)

        i += n_combs
        total_proba += proba

    return i


def generate_binary_mapping(n_original_bits, n_mapped_bits: int, stop_at_i: int) -> Dict[tuple, tuple]:
    orig_truth_table = gen_truth_table(n_original_bits)
    compressed_truth_table = gen_truth_table(n_mapped_bits)
    mapping = {}
    for i, item in enumerate(orig_truth_table):
        item_binary = tuple(np.array(item).astype(int))
        try:
            compressed_binary = tuple(np.array(next(compressed_truth_table)).astype(int))
        except StopIteration:
            break
        mapping[item_binary] = compressed_binary
        if i == stop_at_i:
            break
    return mapping


class Compressor:
    def __init__(self, n_bits: int, one_proba: float, error_proba: float, block_size: int):
        self.n_bits = n_bits
        self.one_proba = one_proba
        self.error_proba = error_proba
        self.block_size = block_size

        assert n_bits % block_size == 0

        self.subset_cutoff = get_subset_cutoff(self.block_size, self.one_proba, self.error_proba)
        self.n_compressed_bits = int(np.ceil(np.log2(self.subset_cutoff)))
        self.lookup_table = generate_binary_mapping(self.block_size, self.n_compressed_bits, stop_at_i=self.subset_cutoff)

    def compress(self, raw_binary_string: NDArray[np.int_]) -> NDArray[np.int_]:
        new_string = []
        for i in range(0, len(raw_binary_string), self.block_size):
            block = raw_binary_string[i:i+self.block_size]
            compressed_block = np.array(self.lookup_table[tuple(block)])
            new_string.append(compressed_block)
        return np.concatenate(new_string)

    def decompress(self, compressed_binary_string: NDArray[np.int_]) -> NDArray[np.int_]:
        reverse_lookup = {v: k for k, v in self.lookup_table.items()}
        new_string = []
        for i in range(0, len(compressed_binary_string), self.n_compressed_bits):
            block = compressed_binary_string[i:i+ self.n_compressed_bits]
            uncompressed_block = np.array(reverse_lookup[tuple(block)])
            new_string.append(uncompressed_block)
        return np.concatenate(new_string)
