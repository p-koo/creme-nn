import os
import numpy as np 
import tensorflow as tf
import tensorflow_hub as hub
import glob
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
        self.target_length = 896
        if type(self.bin_index)==int:
            self.bin_index = [self.bin_index]
        if type(self.track_index)==int:
            self.track_index = [self.track_index]


    def predict(self, x, batch_size=1):
        """Get full predictions from enformer in batches."""

        # check to make sure shape is correct
        if len(x.shape) == 2:
            x = x[np.newaxis]

        # get predictions
        if x.shape[1] == self.pseudo_pad:
            x = np.pad(x, ((0, 0), (self.pseudo_pad // 2, self.pseudo_pad // 2), (0, 0)), 'constant')
        N = x.shape[0]

        # get predictions
        if batch_size < N:
            preds = []
            i = 0
            for batch in batch_np(x, batch_size):
                preds.append(self.model.predict_on_batch(batch)[self.head].numpy())
                i += batch.shape[0]
            preds = np.concatenate(preds)
        else:
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
            print('Add axis')
            x = x[np.newaxis]
        assert x.shape[1] == self.pseudo_pad + self.seq_length, 'Bad seq length'
            # print('Add pads')
            # print(type(x))
            # x = self.pad_seq(x)
            # x = np.pad(x, ((0, 0), (self.pseudo_pad // 2, self.pseudo_pad // 2), (0, 0)), 'constant')

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

