# coding=utf-8
import itertools
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.backend import batch_dot
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import TruncatedNormal

import tensorflow as tf


class FEFMLayer(Layer):
    """Field-Embedded Factorization Machines

      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape:
            ``(batch_size, (num_fields * (num_fields-1))/2)`` # concatenated FEFM interaction embeddings

      Arguments
        - **regularizer** : L2 regularizer weight for the field pair matrix embeddings parameters of FEFM

      References
        - [Field-Embedded Factorization Machines for Click-through Rate Prediction]
         https://arxiv.org/pdf/2009.09931.pdf
    """

    def __init__(self, regularizer, **kwargs):
        self.regularizer = regularizer
        super(FEFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                                expect to be 3 dimensions" % (len(input_shape)))

        self.num_fields = int(input_shape[1])
        embedding_size = int(input_shape[2])

        self.field_embeddings = {}
        for fi, fj in itertools.combinations(range(self.num_fields), 2):
            field_pair_id = str(fi) + "-" + str(fj)
            self.field_embeddings[field_pair_id] = self.add_weight(name='field_embeddings' + field_pair_id,
                                                                   shape=(embedding_size, embedding_size),
                                                                   initializer=TruncatedNormal(),
                                                                   regularizer=l2(self.regularizer),
                                                                   trainable=True)

        super(FEFMLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))

        pairwise_inner_prods = []
        for fi, fj in itertools.combinations(range(self.num_fields), 2):
            field_pair_id = str(fi) + "-" + str(fj)
            feat_embed_i = tf.squeeze(inputs[0:, fi:fi + 1, 0:], axis=1)
            feat_embed_j = tf.squeeze(inputs[0:, fj:fj + 1, 0:], axis=1)
            field_pair_embed_ij = self.field_embeddings[field_pair_id]

            feat_embed_i_tr = tf.matmul(feat_embed_i, field_pair_embed_ij + tf.transpose(field_pair_embed_ij))

            f = batch_dot(feat_embed_i_tr, feat_embed_j, axes=1)
            pairwise_inner_prods.append(f)

        concat_vec = tf.concat(pairwise_inner_prods, axis=1)
        return concat_vec

    def compute_output_shape(self, input_shape):
        num_fields = int(input_shape[1])
        return (None, (num_fields * (num_fields - 1)) / 2)

    def get_config(self):
        config = super(FEFMLayer, self).get_config().copy()
        config.update({
            'regularizer': self.regularizer,
        })
        return config
