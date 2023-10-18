import os 
import numpy as np 
import tensorflow as tf
import tensorflow_hub as hub

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
    def __init__(self, head='human', track_index=5111):

        # path to enformer on tensorflow-hub
        tfhub_url = 'https://tfhub.dev/deepmind/enformer/1'
        os.environ['TFHUB_CACHE_DIR'] = '.'
        self.model = hub.load(tfhub_url).model
        self.head = head
        self.track_index = track_index
        self.pseudo_pad = 196608


    def predict_on_batch(self, x):
        """Get full predictions from Enformer."""
        assert len(x.shape) == 3, 'input not 3D'
        # Enformer uses 196608 extra input length which does not affect the predictions
        if x.shape[1] == self.pseudo_pad:
            x = np.pad(x, ((0, 0), (self.pseudo_pad // 2, self.pseudo_pad // 2), (0, 0)), 'constant')
        predictions = self.model.predict_on_batch(x)
        return {k: v.numpy() for k, v in predictions.items()}


    def predict(self, x, batch_size=1):
        """Get curated predictions from enformer in batches."""

        # check to make sure shape is correct
        if len(x.shape) == 2:
            x = x[np.newaxis]
        N = x.shape[0]

        # get predictions
        if batch_size < N:
            preds = []
            i = 0
            for batch in batch_np(x, batch_size):
                preds.append(self.predict_on_batch(batch)[self.head][:,:,self.track_index])
                i += batch.shape[0]
            return np.array(preds)
        else:
            return self.predict_on_batch(x)[self.head][:, :, self.track_index]


    def predict_all(self, x, batch_size=1):
        """Get full predictions from enformer in batches."""

        # check to make sure shape is correct
        if len(x.shape) == 2:
            x = x[np.newaxis]
        N = x.shape[0]

        # get predictions    
        if batch_size < N:
            preds = []
            i = 0
            for batch in batch_np(x, batch_size):
                preds.append(self.predict_on_batch(batch)[self.head])
                i += batch.shape[0]
            return np.array(preds)
        else:
            return self.predict_on_batch(x)[self.head]


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

