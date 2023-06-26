import os
import pyfaidx
import kipoiseq


def make_dir(dir_path):
    """
    :param dir_path: new directory path
    :return: str path
    """
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return dir_path



"""The fasta string extractor for enformer"""
class FastaStringExtractor:
    
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


def one_hot_encode(sequence):
  return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)



def set_tile_range(L, window, stride):
    tiles = []
    i = 0
    while i < L:
        tiles.append([i, i+window])
        i += stride 
        if i+window > L:
            break
    return tiles

def remove_tss_tile(tiles, tile_index):
    del tiles[tile_index]



