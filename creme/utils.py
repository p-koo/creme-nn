import os
import pyfaidx
import kipoiseq
import pickle
import numpy as np



def rc_dna(seq):
    """
    Reverse complement the DNA sequence
    >>> assert rc_seq("TATCG") == "CGATA"
    >>> assert rc_seq("tatcg") == "cgata"
    """
    rc_hash = {
        "A": "T",
        "T": "A",
        "C": "G",
        "G": "C",
        "N": "N"

    }
    return "".join([rc_hash[s.upper()] for s in seq[::-1]])

########################################################################################
# Classes
########################################################################################

class SequenceParser():
    """Sequence parser from fasta file for enformer."""

    def __init__(self, fasta_path):
        self.fasta_extractor = FastaStringExtractor(fasta_path) 

    def extract_seq_centered(self, chrom, midpoint, strand, seq_len, onehot=True):
        assert strand in ['+', '-'], 'bad strand!'
        # get coordinates for tss
        target_interval = kipoiseq.Interval(chrom, midpoint, midpoint+1).resize(seq_len)

        # get sequence from reference genome
        seq = self.fasta_extractor.extract(target_interval)
        if strand == '-':
            seq = rc_dna(seq)
        if onehot:
            return one_hot_encode(seq)
        else:
            return seq

    def extract_seq_interval(self, chrom, start, end, strand, seq_len=None, onehot=True):
        assert strand in ['+', '-'], 'bad strand!'
        # get coordinates for tss
        target_interval = kipoiseq.Interval(chrom, start, end)

        if seq_len:
            target_interval = target_interval.resize(seq_len)

        # get sequence from reference genome
        seq = self.fasta_extractor.extract(target_interval)
        if strand == '-':
            seq = rc_dna(seq)
        if onehot:
            return one_hot_encode(seq)
        else:
            return seq


class FastaStringExtractor:
    """Fasta string extractor for enformer."""

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
    """Convert sequence to one-hot."""
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


def set_tile_range(L, tile_size):
    """Create tile coordinates for input sequence."""

    # get center tile

    tss_coordinate = L // 2
    downstream = sorted(list(range(tss_coordinate + tile_size // 2, L, tile_size)))
    upstream = sorted(list(range(tss_coordinate - tile_size // 2, 0, -tile_size)))

    perturb_range = upstream + downstream
    return perturb_range

def make_dir(dir_path):
    """ Make directory if doesn't exist."""
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return dir_path

def get_summary(row):
    return f"{row['gene_name']}_{row['Chromosome']}_{row['Start']}_{row['Strand']}"

def plot_cdf(x, bins=1000):
    # getting data of the histogram
    count, bins_count = np.histogram(x, bins=bins)

    # finding the PDF of the histogram using count values
    pdf = count / sum(count)

    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)

    # plotting PDF and CDF

    return cdf

def clean_cell_name(t):
     return t.split(':')[-1].split(' ENCODE')[0].strip()

def save_pickle(result_path, x):
    with open(result_path, 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(result_path):
    with open(result_path, 'rb') as handle:
        context_res = pickle.load(handle)
    return context_res