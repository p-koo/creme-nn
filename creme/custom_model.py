import os
import numpy as np 
import tensorflow as tf
import tensorflow_hub as hub
import glob
import seqnn
import json
import pandas as pd

########################################################################################
# CREME model
########################################################################################


class ModelBase():
    def __init__(self):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

########################################################################################
# Borzoi model
########################################################################################




class Borzoi(ModelBase):
    """
    Wrapper class for Borzoi.
    inputs:
        head : str
            Borzoi head to get predictions --> head or mouse.
        track_index : int
            Enformer index of prediciton track for a given head.
    """
    def __init__(self, model_path, track_index=None, bin_index=None, cage_tracks=None, rna_tracks=None, aggregate=False, params_file='../data/borzoi_params_pred.json',
                 targets_file='../data/borzoi_targets_human.txt'):

        # Read model parameters
        with open(params_file) as params_open:
            params = json.load(params_open)
            params_model = params['model']
            params_model['norm_type'] = 'batch' # makes compatible with 2.11 tf and doesn't change output
        self.seq_length = params_model['seq_length']
        self.models = []
        self.track_index = track_index
        self.bin_index = bin_index
        self.aggregate = aggregate

        targets_df = pd.read_csv(targets_file, index_col=0, sep='\t')
        target_index = targets_df.index
        print('Adding models:')
        print(glob.glob(model_path))
        for one_model_path in glob.glob(model_path):

            seqnn_model = seqnn.SeqNN(params_model)
            seqnn_model.restore(one_model_path, 0)
            seqnn_model.build_slice(target_index)
            seqnn_model.build_ensemble(False, '0')
            self.models.append(seqnn_model)

        self.target_crops = seqnn_model.target_crops[0]
        self.model_strides = seqnn_model.model_strides[0]
        self.target_lengths = seqnn_model.target_lengths[0]

        self.rna_tracks = rna_tracks
        self.cage_tracks = cage_tracks


    def predict(self, x):
        if len(x.shape) == 2:
            x = x[np.newaxis]
        preds = []
        for j, m in enumerate(self.models):
            preds.append(m(x)[:, None, ...].astype("float16"))
        preds = np.concatenate(preds, axis=1)

        if self.bin_index:
            preds = preds[:, :, self.bin_index, :]
        if self.track_index:
            preds = preds[..., self.track_index]
        if self.aggregate:
            preds = preds.mean(axis=1)
        return preds

    def predict_cage_rna(self, x, gene, center_pos, cage_bins=None, return_exons=True, return_all=False):
        """Get full predictions from borzoi in batches."""
        # get exon information
        start = center_pos - self.seq_length // 2
        end = center_pos + self.seq_length // 2
        seq_out_start = start + self.model_strides * self.target_crops
        seq_out_len = self.model_strides * self.target_lengths
        if return_exons:
            gene_slice = gene.output_slice(seq_out_start, seq_out_len, self.model_strides, False)
            if len(gene_slice) == 0:
                return False # don't keep results if no exons found!

        # check to make sure shape is correct
        if len(x.shape) == 2:
            x = x[np.newaxis]
        # get predictions
        preds = []
        for j, m in enumerate(self.models):
            preds.append(m(x)[:, None, ...].astype("float16"))
        preds = np.concatenate(preds, axis=1)
        if gene.strand == '-':
            preds = preds[:, :, ::-1, :]  # flip predictions to return to + strand coordinates of exons
        pred_subset = {}
        if self.cage_tracks:
            cage_preds = preds[..., self.cage_tracks]
            if cage_bins is not None:
                assert len(cage_bins) > 0, 'Bins requested but list empty'
                cage_preds = cage_preds[:, :, cage_bins, :]
            pred_subset['cage'] = cage_preds
        if self.rna_tracks:
            rna_preds = preds[..., self.rna_tracks]
            if return_exons:
                rna_preds = rna_preds[:, :, gene_slice, :]
            pred_subset['rna'] = rna_preds

        if return_all:
            pred_subset['all'] = preds

        if self.aggregate:
            for k, v in pred_subset.items():
                pred_subset[k] = v.mean(axis=1)


        return pred_subset

    def track_sequence(self, sequence):
        """Track pooling, striding, and cropping of sequence.

        Args:
          sequence (tf.Tensor): Sequence input.
        """
        self.model_strides = []
        self.target_lengths = []
        self.target_crops = []
        for model in self.models:
            # determine model stride
            self.model_strides.append(1)
            for layer in self.model.layers:
                if hasattr(layer, "strides") or hasattr(layer, "size"):
                    stride_factor = layer.input_shape[1] / layer.output_shape[1]
                    self.model_strides[-1] *= stride_factor
            self.model_strides[-1] = int(self.model_strides[-1])

            # determine predictions length before cropping
            if type(sequence.shape[1]) == tf.compat.v1.Dimension:
                target_full_length = sequence.shape[1].value // self.model_strides[-1]
            else:
                target_full_length = sequence.shape[1] // self.model_strides[-1]

            # determine predictions length after cropping
            self.target_lengths.append(model.outputs[0].shape[1])
            if type(self.target_lengths[-1]) == tf.compat.v1.Dimension:
                self.target_lengths[-1] = self.target_lengths[-1].value
            self.target_crops.append(
                (target_full_length - self.target_lengths[-1]) // 2
            )

        if self.verbose:
            print("model_strides", self.model_strides)
            print("target_lengths", self.target_lengths)
            print("target_crops", self.target_crops)




########################################################################################
# Enformer model
########################################################################################


class Enformer(ModelBase):
    """ 
    Wrapper class for Enformer. 
    inputs:
        head : str 
            Enformer head to get predictions --> head or mouse.
        track_index : int
            Enformer index of prediciton track for a given head.
    """
    def __init__(self, track_index=None, bin_index=None, head='human'):

        # path to enformer on tensorflow-hub
        tfhub_url = 'https://tfhub.dev/deepmind/enformer/1'
        os.environ['TFHUB_CACHE_DIR'] = '.'
        self.model = hub.load(tfhub_url).model
        self.head = head
        self.track_index = track_index
        self.bin_index = bin_index
        self.seq_length = 196608
        self.pseudo_pad = 196608
        if type(self.bin_index)==int:
            self.bin_index = [self.bin_index]
        if type(self.track_index)==int:
            self.track_index = [self.track_index]


    def predict(self, x):
        """Get full predictions from enformer in batches."""

        # check to make sure shape is correct
        if len(x.shape) == 2:
            x = x[np.newaxis]

        # get predictions
        if x.shape[1] == self.pseudo_pad:
            x = np.pad(x, ((0, 0), (self.pseudo_pad // 2, self.pseudo_pad // 2), (0, 0)), 'constant')
        preds = self.model.predict_on_batch(x)[self.head].numpy()
        if self.bin_index:
            preds = preds[:, self.bin_index, :]
        if self.track_index:
            preds = preds[..., self.track_index]
        return preds


    @tf.function
    def contribution_input_grad(self, x, target_mask, head='human', mult_by_input=True):
        """Calculate input gradients"""

        # check to make sure shape is correct
        if len(x.shape) == 2:
            x = x[np.newaxis]

        # calculate saliency maps
        target_mask_mass = tf.reduce_sum(target_mask)
        with tf.GradientTape() as tape:
            tape.watch(x)
            prediction = tf.reduce_sum(
                target_mask[tf.newaxis] * self.model.predict_on_batch(x)[head]
                ) / target_mask_mass
        input_grad = tape.gradient(prediction, x)

        # process saliency maps
        if mult_by_input:
            input_grad *= x
            input_grad = tf.squeeze(input_grad, axis=0)
            return tf.reduce_sum(input_grad, axis=-1)
        else:
            return input_grad



########################################################################################
# Template for custom model 
########################################################################################

# class CustomModel(ModelBase):
#   def __init__(self, model):
#       self.model = model
#   def predict(self, x, class_index, batch_size=64):
#       # insert custom code here to get model predictions
#       # preds = model.predict(x, batch_size=64)
#       # preds = preds[:, class_index]
#       return preds



########################################################################################
# useful functions
########################################################################################


def batch_np(whole_dataset, batch_size):
    """Batch generator for dataset."""
    for i in range(0, whole_dataset.shape[0], batch_size):
        yield whole_dataset[i:i + batch_size]

