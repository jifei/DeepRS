# coding=utf-8
from collections import namedtuple
from tensorflow.keras.initializers import RandomNormal

DEFAULT_GROUP_NAME = "default_group"


class SparseFeature(namedtuple('SparseFeature',
                               ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype',
                                'embeddings_initializer', 'embedding_name', 'group_name', 'trainable'])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embeddings_initializer=None,
                embedding_name=None, group_name=DEFAULT_GROUP_NAME, trainable=True):

        if embedding_dim is None:
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if embeddings_initializer is None:
            embeddings_initializer = RandomNormal(mean=0.0, stddev=0.0001, seed=2020)

        if embedding_name is None:
            embedding_name = name

        return super(SparseFeature, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                                 embeddings_initializer, embedding_name, group_name, trainable)

    def __hash__(self):
        return self.name.__hash__()


class VarLenSparseFeature(namedtuple('VarLenSparseFeature',
                                     ['sparse_feature', 'max_length', 'combiner', 'length_name', 'weight_name',
                                      'weight_norm'])):
    __slots__ = ()

    def __new__(cls, sparsefeat, max_length, combiner="mean", length_name=None, weight_name=None, weight_norm=True):
        return super(VarLenSparseFeature, cls).__new__(cls, sparsefeat, max_length, combiner, length_name, weight_name,
                                                       weight_norm)

    @property
    def name(self):
        return self.sparse_feature.name

    @property
    def vocabulary_size(self):
        return self.sparse_feature.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparse_feature.embedding_dim

    @property
    def use_hash(self):
        return self.sparse_feature.use_hash

    @property
    def dtype(self):
        return self.sparse_feature.dtype

    @property
    def embeddings_initializer(self):
        return self.sparse_feature.embeddings_initializer

    @property
    def embedding_name(self):
        return self.sparse_feature.embedding_name

    @property
    def group_name(self):
        return self.sparse_feature.group_name

    @property
    def trainable(self):
        return self.sparse_feature.trainable

    def __hash__(self):
        return self.name.__hash__()


class DenseFeature(namedtuple('DenseFeature', ['name', 'dimension', 'dtype', 'transform_fn'])):
    """ Dense feature
    Args:
        name: feature name,
        dimension: dimension of the feature, default = 1.
        dtype: dtype of the feature, default="float32".
        transform_fn: If not `None` , a function that can be used to transform
        values of the feature.  the function takes the input Tensor as its
        argument, and returns the output Tensor.
        (e.g. lambda x: (x - 3.0) / 4.2).
    """
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32", transform_fn=None):
        return super(DenseFeature, cls).__new__(cls, name, dimension, dtype, transform_fn)

    def __hash__(self):
        return self.name.__hash__()
