# coding=utf-8
"""
 @author:jifei@outlook.com
 @date  :2021/11/3 21:28
"""

import tensorflow as tf
from ...layers.fm import FM
from ...layers.linear import Linear
from ...layers.dnn import DNN
from ...layers.Inputs import input_embedding_layer
from tensorflow.keras.layers import Input, Flatten, Concatenate, Dense
from tensorflow.keras.initializers import Zeros, glorot_normal

from tensorflow.keras.models import Model

def DeepFM(sparse_features,
           sparse_feature_reindex_dict,
           dense_features,
           dnn_hidden_units=(256, 128, 64),
           l2_reg_embedding=1e-5,
           l2_reg_linear=1e-5,
           l2_reg_dnn=0,
           init_std=0.0001,
           seed=1024,
           bi_dropout=0.2,
           dnn_dropout=0,
           dnn_activation='relu',
           embedding_dim=4):
    # 1. Input & Embedding sparse features
    sparse_input_layer_list = []
    sparse_embedding_layer_list = []
    linear_sparse_embedding_layer_list = []
    for i in sparse_features:
        shape = 1
        name = i
        vocabulary_size = len(sparse_feature_reindex_dict[i]) + 1
        cur_sparse_feature_input_layer, cur_sparse_feature_embedding_layer, cur_linear_sparse_embedding_layer_list = input_embedding_layer(
            shape=shape,
            name=name,
            vocabulary_size=vocabulary_size,
            embedding_dim=embedding_dim,
            l2_reg=l2_reg_embedding
        )

        sparse_input_layer_list.append(cur_sparse_feature_input_layer)
        sparse_embedding_layer_list.append(cur_sparse_feature_embedding_layer)
        linear_sparse_embedding_layer_list.append(cur_linear_sparse_embedding_layer_list)

    # 2. Input dense features
    dense_input_layer_list = []
    for j in dense_features:
        dense_input_layer_list.append(Input(shape=(1,), name=j))

    # Linear part
    sparse_linear_embedding_input = Concatenate(axis=-1)(linear_sparse_embedding_layer_list)
    dense_linear_input = Concatenate(axis=-1)(dense_input_layer_list)
    linear_logit = Linear()([sparse_linear_embedding_input, dense_linear_input])

    # FM part
    fm_logit = FM()(Concatenate(axis=1)(sparse_embedding_layer_list))

    # DNN part
    sparse_linear_input = Concatenate(axis=-1)(sparse_embedding_layer_list)
    dnn_input = Concatenate(axis=-1)([Flatten()(sparse_linear_input), dense_linear_input])
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed)(dnn_input)
    dnn_logit = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=glorot_normal(seed=seed))(dnn_output)

    # output
    out = Dense(1, activation='sigmoid')(tf.keras.layers.add([linear_logit, fm_logit, dnn_logit]))
    return Model(inputs=sparse_input_layer_list + dense_input_layer_list, outputs=out, name='DeepFM')
