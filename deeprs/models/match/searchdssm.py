# coding=utf-8
"""
 @author:jifei
 @date  :2021/11/19 16:10
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, LSTM, Bidirectional, Flatten
from itertools import chain
from ...layers.dnn import DNN
from ...features.inputs import build_input_features, input_from_feature_columns, create_embedding_matrix


def DSSM(query_feature_columns, doc_feature_columns, query_dnn_hidden_units=(64, 32),
         doc_dnn_hidden_units=(64, 32),
         dnn_activation='relu', dnn_use_bn=False,
         l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, seed=1024):
    """Instantiates the Deep Structured Semantic Model architecture.
    :param query_feature_columns: An iterable containing query's features used by  the model.
    :param doc_feature_columns: An iterable containing doc's features used by  the model.
    :param query_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of query tower
    :param doc_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of doc tower
    :param dnn_activation: Activation function to use in deep net
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param seed: integer ,to use as random seed.
    :param metric: str, ``"cos"`` for  cosine  or  ``"ip"`` for inner product
    :return: A Keras model instance.
    """
    # query tower
    query_features = build_input_features(query_feature_columns)
    query_inputs_list = list(query_features.values())
    query_sparse_embedding_dict, query_dense_value_list = input_from_feature_columns(query_features,
                                                                                     query_feature_columns,
                                                                                     l2_reg_embedding)
    query_dnn_input = query_sparse_embedding_dict.values()
    query_dnn_input = list(chain.from_iterable(query_dnn_input))[0]
    # query_dnn_input = Bidirectional(LSTM(64, return_sequences=True))(query_dnn_input)
    query_dnn_input = LSTM(64, return_sequences=True)(query_dnn_input)
    # query_dnn_input = Flatten()(query_dnn_input)
    query_dnn_out = DNN(query_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                        dnn_use_bn, seed=seed)(query_dnn_input)

    # doc tower
    doc_features = build_input_features(doc_feature_columns)
    doc_inputs_list = list(doc_features.values())
    doc_sparse_embedding_dict, doc_dense_value_list = input_from_feature_columns(doc_features,
                                                                                 doc_feature_columns,
                                                                                 l2_reg_embedding)
    doc_dnn_input = list(chain.from_iterable(doc_sparse_embedding_dict.values()))[0]
    # doc_dnn_input = Bidirectional(LSTM(64, return_sequences=True))(doc_dnn_input)
    doc_dnn_input = LSTM(64, return_sequences=True)(doc_dnn_input)
    # doc_dnn_input = Flatten()(doc_dnn_input)
    doc_dnn_out = DNN(doc_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                      dnn_use_bn, seed=seed)(doc_dnn_input)
    query_dnn_out = tf.nn.l2_normalize(query_dnn_out, axis=-1)  # L2归一化
    doc_dnn_out = tf.nn.l2_normalize(doc_dnn_out, axis=-1)  # L2归一化
    score = tf.reduce_sum(query_dnn_out * doc_dnn_out, axis=1, keepdims=True)  # 点积
    output = Dense(1, activation='sigmoid', name='out')(score)
    model = Model(inputs=query_inputs_list + doc_inputs_list, outputs=output)

    model.__setattr__("query_input", query_inputs_list)
    model.__setattr__("doc_input", doc_inputs_list)
    model.__setattr__("query_embedding", query_dnn_out)
    model.__setattr__("doc_embedding", doc_dnn_out)

    return model
