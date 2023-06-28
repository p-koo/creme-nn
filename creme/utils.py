import os
import pyfaidx
import kipoiseq
import numpy as np


########################################################################################
# Classes
########################################################################################

class SequenceParser():
    """sequence parser from fasta file for enformer"""

    def __init__(self, fasta_path):
        self.fasta_extractor = FastaStringExtractor(fasta_path) 

    def extract_seq_centered(self, chrom, midpoint, seq_len, onehot=True):

        # get coordinates for tss
        target_interval = kipoiseq.Interval(chrom, midpoint, midpoint + 1).resize(seq_len)

        # get seequence from reference genome
        seq = self.fasta_extractor.extract(target_interval)

        if onehot:
            return one_hot_encode(seq)
        else:
            return seq

    def extract_seq_interval(self, chrom, start, end, seq_len=None, onehot=True):

        # get coordinates for tss
        target_interval = kipoiseq.Interval(chrom, start, end)

        if seq_len:
            target_interval = target_interval.resize(seq_len)

        # get seequence from reference genome
        seq = self.fasta_extractor.extract(target_interval)

        if onehot:
            return one_hot_encode(seq)
        else:
            return seq


class FastaStringExtractor:
    """The fasta string extractor for enformer"""

    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: kipoiseq.Interval, safe_mode=True, **kwargs) -> str:
        chromosome_length = self._chromosome_sizes[interval.chrom]
        # if interval is completely outside chromosome boundaries ...
        if (interval.start < 0 and interval.end < 0 or
            interval.start >= chromosome_length and interval.end > chromosome_length):
            if safe_mode:  # if safe mode is on: fail
                raise ValueError("Interval outside chromosome boundaries")
            else:  # if it's off: return N-sequence
                return interval.width() * 'N'
        # ... else interval (at least!) overlaps chromosome boundaries
        else:
            # Truncate interval if it extends beyond the chromosome lengths.
            trimmed_interval = kipoiseq.Interval(interval.chrom,
                                        max(interval.start, 0),
                                        min(interval.end, chromosome_length),
                                        )
            # pyfaidx wants a 1-based interval
            sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                            trimmed_interval.start + 1,
                                            trimmed_interval.stop).seq).upper()
            # Fill truncated values with N's.
            pad_upstream = 'N' * max(-interval.start, 0)
            pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
            return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()



########################################################################################
# Functions
########################################################################################

def one_hot_encode(sequence):
    """convert sequence to one-hot"""
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


def set_tile_range(L, window, stride):
    """create tile coordinates for input sequence"""

    midpoint = int(L/2)
    center_start = midpoint - window//2
    center_end = center_start + window
    center_tile = [center_start, center_end]

    other_tiles = []
    start = np.mod(center_start, window)
    for i in np.arange(start, center_start, window):
        other_tiles.append([i, i+window])
    for i in np.arange(center_end, L, window):
        if i + window < L:
            other_tiles.append([i, i+window])

    return center_tile, other_tiles 


def remove_tss_tile(tiles, tile_index):
    """remove a tile form a list of tile coordinates"""
    del tiles[tile_index]


def make_dir(dir_path):
    """
    :param dir_path: new directory path
    :return: str path
    """
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return dir_path


