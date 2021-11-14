# -*- coding:utf-8 -*-
"""
Author:
    Harshit Pande

Reference:
    [1] Field-Embedded Factorization Machines for Click-through Rate Prediction]
    (https://arxiv.org/pdf/2009.09931.pdf)

    this file also supports all the possible Ablation studies for reproducibility

"""

from itertools import chain

import tensorflow as tf

from ...features.inputs import build_input_features, input_from_feature_columns
from ...features.feature_column import DEFAULT_GROUP_NAME
from ...layers.fefm import FEFMLayer
from ...layers.linear import Linear
from tensorflow.keras.layers import Flatten, Concatenate, Dense
from ...layers.dnn import DNN


def DeepFEFM(linear_feature_columns, dnn_feature_columns, fm_group=(DEFAULT_GROUP_NAME,), use_fefm=True,
             dnn_hidden_units=(256, 128, 64), l2_reg_linear=1e-5, l2_reg_embedding_feat=1e-5,
             l2_reg_embedding_field=1e-5, l2_reg_dnn=0, seed=1024, dnn_dropout=0.0,
             exclude_feature_embed_in_dnn=False,
             use_linear=True, use_fefm_embed_in_dnn=True, dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """Instantiates the DeepFEFM Network architecture or the shallow FEFM architecture (Ablation studies supported)

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param fm_group: list, group_name of features that will be used to do feature interactions.
    :param use_fefm: bool,use FEFM logit or not (doesn't effect FEFM embeddings in DNN, controls only the use of final FEFM logit)
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding_feat: float. L2 regularizer strength applied to embedding vector of features
    :param l2_reg_embedding_field: float, L2 regularizer to field embeddings
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param exclude_feature_embed_in_dnn: bool, used in ablation studies for removing feature embeddings in DNN
    :param use_linear: bool, used in ablation studies
    :param use_fefm_embed_in_dnn: bool, True if FEFM interaction embeddings are to be used in FEFM (set False for Ablation)
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    features = build_input_features(linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    # linear_logit = get_linear_logit(features, linear_feature_columns, l2_reg=l2_reg_linear, seed=seed, prefix='linear')

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                        l2_reg_embedding_feat)

    sparse_embedding_layer_list = list(chain.from_iterable(group_embedding_dict.values()))
    dense_linear_input = Concatenate(axis=-1)(dense_value_list)
    sparse_linear_input = Concatenate(axis=-1)(sparse_embedding_layer_list)
    fm_embedding_layer_list = []
    for k, v in group_embedding_dict.items():
        if k in fm_group:
            fm_embedding_layer_list += v
    # linear part
    linear_logit = Linear(l2_reg=l2_reg_linear)([sparse_linear_input, dense_linear_input])

    # fefm part
    fefm_interaction_embedding = FEFMLayer(regularizer=l2_reg_embedding_field)(
        Concatenate(axis=1)(fm_embedding_layer_list))
    # fefm_interaction_embedding = FEFMLayer(regularizer=l2_reg_embedding_field)(
    #     tf.keras.layers.LayerNormalization(axis=-1)(Concatenate(axis=1)(fm_embedding_layer_list)))
    dnn_input = Concatenate(axis=-1)([Flatten()(sparse_linear_input), dense_linear_input])

    # if use_fefm_embed_in_dnn is set to False it is Ablation4 (Use false only for Ablation)
    if use_fefm_embed_in_dnn:
        if exclude_feature_embed_in_dnn:
            # Ablation3: remove feature vector embeddings from the DNN input
            dnn_input = fefm_interaction_embedding
        else:
            # No ablation
            dnn_input = Concatenate(axis=1)([dnn_input, fefm_interaction_embedding])

    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)

    dnn_logit = Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(dnn_out)

    fefm_logit = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims=True))(fefm_interaction_embedding)

    if len(dnn_hidden_units) == 0 and use_fefm is False and use_linear is True:  # only linear
        final_logit = linear_logit
    elif len(dnn_hidden_units) == 0 and use_fefm is True and use_linear is True:  # linear + FEFM
        final_logit = tf.keras.layers.add([linear_logit, fefm_logit])
    elif len(dnn_hidden_units) > 0 and use_fefm is False and use_linear is True:  # linear +　Deep # Ablation1
        final_logit = tf.keras.layers.add([linear_logit, dnn_logit])
    elif len(dnn_hidden_units) > 0 and use_fefm is True and use_linear is True:  # linear + FEFM + Deep
        final_logit = tf.keras.layers.add([linear_logit, fefm_logit, dnn_logit])
    elif len(dnn_hidden_units) == 0 and use_fefm is True and use_linear is False:  # only FEFM (shallow)
        final_logit = fefm_logit
    elif len(dnn_hidden_units) > 0 and use_fefm is False and use_linear is False:  # only Deep
        final_logit = dnn_logit
    elif len(dnn_hidden_units) > 0 and use_fefm is True and use_linear is False:  # FEFM + Deep # Ablation2
        final_logit = tf.keras.layers.add([fefm_logit, dnn_logit])
    else:
        raise NotImplementedError

    output = Dense(1, activation='sigmoid')(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
