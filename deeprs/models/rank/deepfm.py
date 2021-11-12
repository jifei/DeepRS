# coding=utf-8
"""
 @author:jifei@outlook.com
 @date  :2021/11/3 21:28
"""
from itertools import chain

import tensorflow as tf
from ...layers.fm import FM
from ...layers.linear import Linear
from ...layers.dnn import DNN
from ...features.inputs import build_input_features, input_from_feature_columns
from ...features.feature_column import DEFAULT_GROUP_NAME
from tensorflow.keras.layers import Flatten, Concatenate, Dense

from tensorflow.keras.models import Model


#
# def combined_dnn_input(sparse_embedding_list, dense_value_list):
#     if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
#         sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
#         dense_dnn_input = Flatten()(concat_func(dense_value_list))
#         return concat_func([sparse_dnn_input, dense_dnn_input])
#     elif len(sparse_embedding_list) > 0:
#         return Flatten()(concat_func(sparse_embedding_list))
#     elif len(dense_value_list) > 0:
#         return Flatten()(concat_func(dense_value_list))
#     else:
#         raise NotImplementedError("dnn_feature_columns can not be empty list")
#

def DeepFM(linear_feature_columns, dnn_feature_columns, fm_group=(DEFAULT_GROUP_NAME,), dnn_hidden_units=(256, 128, 64),
           l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_dnn=1e-5, seed=1024, dnn_dropout=0,
           dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param fm_group: list, group_name of features that will be used to do feature interactions.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding)
    sparse_embedding_layer_list = list(chain.from_iterable(group_embedding_dict.values()))
    fm_embedding_layer_list = []
    for k, v in group_embedding_dict.items():
        if k in fm_group:
            fm_embedding_layer_list += v

    dense_linear_input = Concatenate(axis=-1)(dense_value_list)
    sparse_linear_input = Concatenate(axis=-1)(sparse_embedding_layer_list)

    # linear part
    linear_logit = Linear(l2_reg=l2_reg_linear)([sparse_linear_input, dense_linear_input])

    # fm part
    fm_logit = FM()(Concatenate(axis=1)(fm_embedding_layer_list))

    # dnn part
    dnn_input = Concatenate(axis=-1)([Flatten()(sparse_linear_input), dense_linear_input])
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    dnn_logit = Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_output)
    # input and  output
    inputs = list(features.values())
    outputs = Dense(1, activation='sigmoid')(tf.keras.layers.add([linear_logit, fm_logit, dnn_logit]))
    return Model(inputs=inputs, outputs=outputs)
