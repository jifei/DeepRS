# coding=utf-8
from .feature_column import DenseFeature, SparseFeature, VarLenSparseFeature
from itertools import chain
from collections import OrderedDict, defaultdict
from tensorflow.keras.layers import Input, Embedding, Lambda
from tensorflow.python.keras.layers import Hashing
from tensorflow.keras.regularizers import l2
from ..layers.sequence import WeightedSequenceLayer, SequencePoolingLayer


def create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    sparse_embedding = {}
    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            if feat.embedding_name not in sparse_embedding:
                emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                                embeddings_initializer=feat.embeddings_initializer,
                                embeddings_regularizer=l2(
                                    l2_reg),
                                name=prefix + '_seq_emb_' + feat.name,
                                mask_zero=seq_mask_zero)
                emb.trainable = feat.trainable
                sparse_embedding[feat.embedding_name] = emb

    for feat in sparse_feature_columns:
        if feat.embedding_name not in sparse_embedding:
            emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                            embeddings_initializer=feat.embeddings_initializer,
                            embeddings_regularizer=l2(l2_reg),
                            name=prefix + '_emb_' + feat.embedding_name)
            emb.trainable = feat.trainable
            sparse_embedding[feat.embedding_name] = emb


    return sparse_embedding


def get_embedding_vec_list(embedding_dict, input_dict, sparse_feature_columns, return_feat_list=()):
    embedding_vec_list = []
    for fg in sparse_feature_columns:
        feature_name = fg.name
        if len(return_feat_list) == 0 or feature_name in return_feat_list:
            if fg.use_hash:
                lookup_idx = Hashing(fg.vocabulary_size)(input_dict[feature_name])
            else:
                lookup_idx = input_dict[feature_name]

            embedding_vec_list.append(embedding_dict[feature_name](lookup_idx))

    return embedding_vec_list


def create_embedding_matrix(feature_columns, l2_reg, prefix="", seq_mask_zero=True):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeature), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeature), feature_columns)) if feature_columns else []
    sparse_embedding_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, l2_reg,
                                                  prefix=prefix + 'sparse', seq_mask_zero=seq_mask_zero)
    return sparse_embedding_dict


def embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, filter_feature_list=(),
                     mask_feat_list=(), to_list=False):
    group_embedding_dict = defaultdict(list)
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if len(filter_feature_list) == 0 or feature_name in filter_feature_list:
            if fc.use_hash:
                mask_value = 0 if feature_name in mask_feat_list else None
                lookup_idx = Hashing(fc.vocabulary_size, mask_value=mask_value)(sparse_input_dict[feature_name])
            else:
                lookup_idx = sparse_input_dict[feature_name]

            group_embedding_dict[fc.group_name].append(sparse_embedding_dict[embedding_name](lookup_idx))
    if to_list:
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict


def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            lookup_idx = Hashing(fc.vocabulary_size, mask_value=0)(sequence_input_dict[feature_name])
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)
    return varlen_embedding_vec_dict


def get_varlen_pooling_list(embedding_dict, features, varlen_sparse_feature_columns, to_list=False):
    pooling_vec_list = defaultdict(list)
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        combiner = fc.combiner
        feature_length_name = fc.length_name
        if feature_length_name is not None:
            if fc.weight_name is not None:
                seq_input = WeightedSequenceLayer(weight_normalization=fc.weight_norm)(
                    [embedding_dict[feature_name], features[feature_length_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=False)(
                [seq_input, features[feature_length_name]])
        else:
            if fc.weight_name is not None:
                seq_input = WeightedSequenceLayer(weight_normalization=fc.weight_norm, supports_masking=True)(
                    [embedding_dict[feature_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=True)(
                seq_input)
        pooling_vec_list[fc.group_name].append(vec)
    if to_list:
        return chain.from_iterable(pooling_vec_list.values())
    return pooling_vec_list


def get_dense_input(features, feature_columns):
    """

    :param features:
    :param feature_columns:
    :return: []
    """
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeature), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        if fc.transform_fn is None:
            dense_input_list.append(features[fc.name])
        else:
            transform_result = Lambda(fc.transform_fn)(features[fc.name])
            dense_input_list.append(transform_result)
    return dense_input_list


def build_input_features(feature_columns, prefix=''):
    """

    :param feature_columns:
    :param prefix:
    :return: {feature_name:Input}
    """
    input_features = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, SparseFeature):
            input_features[fc.name] = Input(shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, DenseFeature):
            input_features[fc.name] = Input(shape=(fc.dimension,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, VarLenSparseFeature):
            input_features[fc.name] = Input(shape=(fc.max_length,), name=prefix + fc.name, dtype=fc.dtype)
            if fc.weight_name is not None:
                input_features[fc.weight_name] = Input(shape=(fc.max_length, 1), name=prefix + fc.weight_name,
                                                       dtype="float32")
            if fc.length_name is not None:
                input_features[fc.length_name] = Input((1,), name=prefix + fc.length_name, dtype='int32')

        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return input_features


def get_inputs_list(inputs):
    return list(chain(*list(map(lambda x: x.values(), filter(lambda x: x is not None, inputs)))))


def get_feature_names(feature_columns):
    """
    get list of feature name from feature_columns
    :param feature_columns:
    :return: [str]
    """
    features = build_input_features(feature_columns)
    return list(features.keys())


def input_from_feature_columns(features, feature_columns, l2_reg, prefix='', seq_mask_zero=True):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeature), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeature), feature_columns)) if feature_columns else []

    embedding_matrix_dict = create_embedding_matrix(feature_columns, l2_reg, prefix=prefix, seq_mask_zero=seq_mask_zero)
    sparse_embedding_dict = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns)

    dense_value_list = get_dense_input(features, feature_columns)
    # if not support_dense and len(dense_value_list) > 0:
    #     raise ValueError("DenseFeature is not supported in dnn_feature_columns")

    # sequence_embedding dict
    sequence_embedding_dict = varlen_embedding_lookup(embedding_matrix_dict, features, varlen_sparse_feature_columns)
    # sequence pooling embedding dict
    varlen_sparse_embedding_dict = get_varlen_pooling_list(sequence_embedding_dict, features,
                                                           varlen_sparse_feature_columns)
    embedding_dict = merge_dict(sparse_embedding_dict, varlen_sparse_embedding_dict)
    return embedding_dict, dense_value_list


def merge_dict(a, b):
    c = defaultdict(list)
    for k, v in a.items():
        c[k].extend(v)
    for k, v in b.items():
        c[k].extend(v)
    return c
