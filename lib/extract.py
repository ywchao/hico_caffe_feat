#!/usr/bin/env python
"""
Extractor is an image classifier specialization of Net.
"""

import numpy as np

import caffe


class Extractor(caffe.Net):
    """
    Extractor extends Net for image class prediction
    by scaling, center cropping, or oversampling.

    Parameters
    ----------
    image_dims : dimensions to scale input for cropping/sampling.
        Default is to scale to net input size for whole-image crop.
    mean, input_scale, raw_scale, channel_swap: params for
        preprocessing options.
    """
    def __init__(self, model_file, pretrained_file, feature_name, 
                 image_dims=None, mean=None, input_scale=None, 
                 raw_scale=None, channel_swap=None):
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

        # configure pre-processing
        in_ = self.inputs[0]
        self.transformer = caffe.io.Transformer(
            {in_: self.blobs[in_].data.shape})
        self.transformer.set_transpose(in_, (2, 0, 1))
        if mean is not None:
            self.transformer.set_mean(in_, mean)
        if input_scale is not None:
            self.transformer.set_input_scale(in_, input_scale)
        if raw_scale is not None:
            self.transformer.set_raw_scale(in_, raw_scale)
        if channel_swap is not None:
            self.transformer.set_channel_swap(in_, channel_swap)

        self.crop_dims = np.array(self.blobs[in_].data.shape[2:])
        if not image_dims:
            image_dims = self.crop_dims
        self.image_dims = image_dims
        self.feature_name = feature_name

    # def extract(self, inputs, oversample=True):
    def extract(self, inputs, crop_mode='oversample'):
        """
        Predict classification probabilities of inputs.

        Parameters
        ----------
        inputs : iterable of (H x W x K) input ndarrays.
        oversample : boolean
            average predictions across center, corners, and mirrors
            when True (default). Center-only prediction when False.

        Returns
        -------
        predictions: (N x C) ndarray of class probabilities for N images and C
            classes.
        """
        # # Scale to standardize input dimensions.
        # input_ = np.zeros((len(inputs),
        #                    self.image_dims[0],
        #                    self.image_dims[1],
        #                    inputs[0].shape[2]),
        #                   dtype=np.float32)
        # for ix, in_ in enumerate(inputs):
        #     input_[ix] = caffe.io.resize_image(in_, self.image_dims)
        # 
        # if oversample:
        #     # Generate center, corner, and mirrored crops.
        #     input_ = caffe.io.oversample(input_, self.crop_dims)
        # else:
        #     # Take center crop.
        #     center = np.array(self.image_dims) / 2.0
        #     crop = np.tile(center, (1, 2))[0] + np.concatenate([
        #         -self.crop_dims / 2.0,
        #         self.crop_dims / 2.0
        #     ])
        #     input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]

        assert crop_mode == 'oversample' \
            or crop_mode == 'center' \
            or crop_mode == 'full'
        if crop_mode == 'oversample' or crop_mode == 'center':
            input_ = np.zeros((len(inputs),
                               self.image_dims[0],
                               self.image_dims[1],
                               inputs[0].shape[2]),
                              dtype=np.float32)
            for ix, in_ in enumerate(inputs):
                input_[ix] = caffe.io.resize_image(in_, self.image_dims)
            if crop_mode == 'oversample':
                # Generate center, corner, and mirrored crops.
                input_ = caffe.io.oversample(input_, self.crop_dims)
            if crop_mode == 'center':
                # Take center crop.
                center = np.array(self.image_dims) / 2.0
                crop = np.tile(center, (1, 2))[0] + np.concatenate([
                    -self.crop_dims / 2.0,
                    self.crop_dims / 2.0
                ])
                input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]
        if crop_mode == 'full':
            input_ = np.zeros((len(inputs),
                               self.crop_dims[0],
                               self.crop_dims[1],
                               inputs[0].shape[2]),
                              dtype=np.float32)
            for ix, in_ in enumerate(inputs):
                input_[ix] = caffe.io.resize_image(in_, self.crop_dims)

        # Classify
        caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]],
                            dtype=np.float32)
        for ix, in_ in enumerate(input_):
            caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)
        out = self.forward_all(**{self.inputs[0]: caffe_in, 'blobs': [self.feature_name]})
        # out = self.forward_all(**{self.inputs[0]: caffe_in})
        # predictions = out[self.outputs[0]]
        # print out
        feature_out = out[self.feature_name]

        # For oversampling, average predictions across crops.
        # if oversample:
        if crop_mode == 'oversample':
            feature_out = feature_out.reshape((len(feature_out) / 10, 10, -1))
            feature_out = feature_out.mean(1)

        # print feature_out
        # print np.shape(feature_out)
        # print "sum is: %d" % np.sum(feature_out)
        return feature_out
