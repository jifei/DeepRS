# coding=utf-8
"""
 @author:jifei
 @date  :2021/11/19 16:10
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate
from itertools import chain
from ...layers.dnn import DNN
from ...features.inputs import build_input_features, input_from_feature_columns, create_embedding_matrix


def DSSM(user_feature_columns, item_feature_columns, user_dnn_hidden_units=(64, 32),
         item_dnn_hidden_units=(64, 32),
         dnn_activation='relu', dnn_use_bn=False,
         l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, seed=1024, metric='cos'):
    """Instantiates the Deep Structured Semantic Model architecture.
    :param user_feature_columns: An iterable containing user's features used by  the model.
    :param item_feature_columns: An iterable containing item's features used by  the model.
    :param user_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of user tower
    :param item_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of item tower
    :param dnn_activation: Activation function to use in deep net
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param seed: integer ,to use as random seed.
    :param metric: str, ``"cos"`` for  cosine  or  ``"ip"`` for inner product
    :return: A Keras model instance.
    """
    # user tower
    user_features = build_input_features(user_feature_columns)
    user_inputs_list = list(user_features.values())
    user_sparse_embedding_dict, user_dense_value_list = input_from_feature_columns(user_features,
                                                                                   user_feature_columns,
                                                                                   l2_reg_embedding)

    # user_dnn_input = Concatenate(axis=-1)(Concatenate(axis=-1)(user_sparse_embedding_list),
    #                                       Concatenate(axis=-1)(user_dense_value_list))

    user_dnn_input = Concatenate(axis=-1)(list(chain.from_iterable(user_sparse_embedding_dict.values())))
    user_dnn_out = DNN(user_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, seed=seed)(user_dnn_input)

    # item tower
    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())
    item_sparse_embedding_dict, item_dense_value_list = input_from_feature_columns(item_features,
                                                                                   item_feature_columns,
                                                                                   l2_reg_embedding)
    # item_dnn_input = Concatenate(axis=-1)(Concatenate(axis=-1)(item_sparse_embedding_list),
    #                                       Concatenate(axis=-1)(item_dense_value_list))

    item_dnn_input = Concatenate(axis=-1)(list(chain.from_iterable(item_sparse_embedding_dict.values())))


    item_dnn_out = DNN(item_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, seed=seed)(item_dnn_input)
    user_dnn_out = tf.nn.l2_normalize(user_dnn_out, axis=-1)  # L2归一化
    item_dnn_out = tf.nn.l2_normalize(item_dnn_out, axis=-1)  # L2归一化

    # score = Similarity(type=metric)([user_dnn_out, item_dnn_out])

    score = tf.reduce_sum(user_dnn_out * item_dnn_out, axis=1, keepdims=True)  # 点积
    output = Dense(1, activation='sigmoid', name='out')(score)
    # print(user_inputs_list)
    # print(item_inputs_list)
    model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)

    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("user_embedding", user_dnn_out)
    model.__setattr__("item_embedding", item_dnn_out)

    return model
