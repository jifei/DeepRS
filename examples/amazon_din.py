# coding=utf-8
"""
 @author:jifei
 @date  :2021/11/18 15:23
"""
import numpy as np
from deeprs.models.rank.din import DIN
from deeprs.features.feature_column import SparseFeature, VarLenSparseFeature, DenseFeature
from deeprs.features.inputs import get_feature_names
import pickle
import pandas as pd


def user_map(x, user):
    return user.get(x['user'], 0)


def item_map(x, item):
    return item.get(x['item'], 0)


def cat_map(x, cat):
    return cat.get(x['cat'], 0)


def pad_list(original, length=100, pad=0):
    if len(original) >= length:
        return original[-length:]
    return original + [pad] * (length - len(original))


def his_map(x, item):
    his = x['his'].split("\002")
    return pad_list([item.get(i, 0) for i in his])


def his_cat_map(x, cat):
    his = x['hist_cat'].split("\002")
    return pad_list([cat.get(i, 0) for i in his])


def process_data(file_path, user, movie, cat):
    data = pd.read_csv(file_path, sep="\t", header=None,
                       names=['label', 'user', 'item', 'cat', 'his', 'hist_cat'])
    print("load data ok")

    data['uidx'] = data.apply(lambda x: user_map(x, user), axis=1)
    data['midx'] = data.apply(lambda x: item_map(x, movie), axis=1)
    data['cidx'] = data.apply(lambda x: cat_map(x, cat), axis=1)
    data['hist_midx'] = data.apply(lambda x: his_map(x, movie), axis=1).apply(np.array)
    data['hist_cidx'] = data.apply(lambda x: his_cat_map(x, cat), axis=1).apply(np.array)
    return data


def get_xy_fd():
    pd.set_option('display.max_columns', None)
    user = pickle.load(open("~/Downloads/data/uid_voc.pkl", "rb"))
    movie = pickle.load(open("~/Downloads/data/mid_voc.pkl", "rb"))
    cat = pickle.load(open("~/Downloads/data/cat_voc.pkl", "rb"))

    max_uidx = max(user.values())
    max_midx = max(movie.values())
    max_cidx = max(cat.values())

    feature_columns = [SparseFeature('uidx', max_uidx + 1, embedding_dim=8), SparseFeature(
        'cidx', max_cidx, embedding_dim=8), SparseFeature('midx', max_midx, embedding_dim=8)]
    feature_columns += [
        VarLenSparseFeature(
            SparseFeature('hist_midx', vocabulary_size=max_midx + 1, embedding_dim=8, embedding_name='midx'),
            max_length=100),
        VarLenSparseFeature(SparseFeature('hist_cidx', max_cidx + 1, embedding_dim=8, embedding_name='cidx'),
                            max_length=100, )]
    behavior_feature_list = ["midx", "cidx"]
    train_data = process_data("~/Downloads/data/local_train_splitByUser", user, movie, cat)
    test_data = process_data("~/Downloads/data/local_test_splitByUser", user, movie, cat)

    train_x = {name: np.array(train_data[name].values.tolist()) for name in get_feature_names(feature_columns)}
    test_x = {name: np.array(test_data[name].values.tolist()) for name in get_feature_names(feature_columns)}
    return train_x, train_data[['label']].values, test_x, test_data[
        ['label']].values, feature_columns, behavior_feature_list


if __name__ == "__main__":
    batch_size = 128
    epochs = 3

    train_x, train_y, test_x, test_y, feature_columns, behavior_feature_list = get_xy_fd()
    model = DIN(feature_columns, behavior_feature_list, dnn_dropout=0.1)
    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy', 'AUC'])
    import tensorflow as tf

    print(model.summary())
    tf.keras.utils.plot_model(model, to_file='./din_model.png', show_shapes=True)

    history = model.fit(train_x, train_y, batch_size=batch_size, verbose=1, epochs=epochs, validation_split=0.1)
    result = model.evaluate(test_x, test_y, batch_size=batch_size)
    print(dict(zip(model.metrics_names, result)))
