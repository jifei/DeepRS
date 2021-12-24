# coding=utf-8
"""
 @author:jifei
 @date  :2021/12/22 15:52
"""
import pandas as pd
from deeprs.models.match.searchdssm import DSSM
from deeprs.features.feature_column import SparseFeature, VarLenSparseFeature
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np


def load_vocab(file_path):
    word_dict = {}
    with open(file_path, encoding='utf8') as f:
        for idx, word in enumerate(f.readlines()):
            word = word.strip()
            word_dict[word] = idx
    return word_dict


if __name__ == '__main__':
    pad = '[PAD]'  # 长度低于max_seq_len，用PAD的进行扩充
    unk = '[UNK]'  # 当字典中没有对应的key时，用UNK代替
    max_seq_len = 50
    batch_size = 128

    word_dict = load_vocab("../deeprs/data/vocab.txt")
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(list(word_dict.keys()), list(word_dict.values())),
        default_value=word_dict[unk])
    test_ds = tf.data.experimental.make_csv_dataset('/Users/jifei/dataset/lcqmc/dev_test.tsv', batch_size=batch_size,
                                                    field_delim="\t", label_name="label")
    iterator = test_ds.as_numpy_iterator()
    print(dict(next(iterator)))

    exit()

    train_ds = tf.data.experimental.make_csv_dataset('/Users/jifei/dataset/lcqmc/test.tsv', batch_size=batch_size,
                                                     field_delim="\t", label_name="label")


    #
    def process_fn(features, label):
        # 按字符分词
        features['text_a'] = tf.strings.unicode_split(features['text_a'], 'UTF-8')
        # 填充补齐
        features['text_a'] = features['text_a'].to_tensor(default_value=pad, shape=(None, max_seq_len))
        # 字典映射成数字
        features['text_a'] = table.lookup(features['text_a'])

        features['text_b'] = tf.strings.unicode_split(features['text_b'], 'UTF-8')
        features['text_b'] = features['text_b'].to_tensor(default_value=pad, shape=(None, max_seq_len))
        features['text_b'] = table.lookup(features['text_b'])
        return features, label


    test_ds = test_ds.map(process_fn)

    train_ds = train_ds.map(process_fn)

    learning_rate = 1e-3
    epochs = 1

    user_features = [VarLenSparseFeature(SparseFeature("text_a", vocabulary_size=len(word_dict), embedding_dim=8),
                                         max_length=max_seq_len, combiner='flatten')]
    item_features = [VarLenSparseFeature(SparseFeature("text_b", vocabulary_size=len(word_dict), embedding_dim=8),
                                         max_length=max_seq_len, combiner='flatten')]
    model = DSSM(user_features, item_features)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="binary_crossentropy",
                  metrics=['acc', 'AUC'])
    print(model.summary())
    history = model.fit(train_ds, epochs=epochs, validation_data=test_ds)
