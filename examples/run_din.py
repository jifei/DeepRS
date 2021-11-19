# coding=utf-8
"""
 @author:jifei
 @date  :2021/11/15 17:23
"""
import numpy as np
import tensorflow as tf
from deeprs.models.rank.din import DIN
from deeprs.features.feature_column import SparseFeature, VarLenSparseFeature, DenseFeature
from deeprs.features.inputs import get_feature_names


def get_xy_fd():
    feature_columns = [SparseFeature('user', 3, embedding_dim=10), SparseFeature(
        'gender', 2, embedding_dim=4), SparseFeature('item_id', 3 + 1, embedding_dim=8),
                       SparseFeature('cate_id', 2 + 1, embedding_dim=4), DenseFeature('pay_score', 1)]
    feature_columns += [
        VarLenSparseFeature(
            SparseFeature('hist_item_id', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item_id'),
            max_length=4, length_name="seq_length"),
        VarLenSparseFeature(SparseFeature('hist_cate_id', 2 + 1, embedding_dim=4, embedding_name='cate_id'),
                            max_length=4, length_name="seq_length")]
    # Notice: History behavior sequence feature name must start with "hist_".
    behavior_feature_list = ["item_id", "cate_id"]
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    cate_id = np.array([1, 2, 2])  # 0 is mask value
    pay_score = np.array([0.1, 0.2, 0.3])

    hist_iid = np.array([[1, 2, 3, 0], [3, 2, 1, 0], [1, 2, 0, 0]])
    hist_cate_id = np.array([[1, 2, 2, 0], [2, 2, 1, 0], [1, 2, 0, 0]])
    seq_length = np.array([3, 3, 2])  # the actual length of the behavior sequence

    feature_dict = {'user': uid, 'gender': ugender, 'item_id': iid, 'cate_id': cate_id,
                    'hist_item_id': hist_iid, 'hist_cate_id': hist_cate_id,
                    'pay_score': pay_score, 'seq_length': seq_length}
    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    y = np.array([1, 0, 1])
    return x, y, feature_columns, behavior_feature_list


if __name__ == "__main__":
    x, y, feature_columns, behavior_feature_list = get_xy_fd()
    model = DIN(feature_columns, behavior_feature_list,dnn_dropout=0.1)
    # model = BST(feature_columns, behavior_feature_list,att_head_num=4)
    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    print(model.summary())
    tf.keras.utils.plot_model(model, to_file='./din_model.png', show_shapes=True)
    history = model.fit(x, y, verbose=1, epochs=10, validation_split=0.5)
