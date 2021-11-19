# coding=utf-8
"""
 @author:jifei
 @date  :2021/11/11 16:27
"""
import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)
from deeprs.models.rank.deepfefm import DeepFEFM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deeprs.features.feature_column import SparseFeature, DenseFeature
from deeprs.features.inputs import get_feature_names
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# 1.set hyperparameters
test_size = 0.2
validation_size = 0.1

batch_size = 256
epochs = 1
hidden_units = (256, 128, 64)
dropout = 0
learning_rate = 1e-3
embedding_dim = 8
l2_reg_dnn = 0

# 2.read data from csv
names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
         'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
         'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
         'C23', 'C24', 'C25', 'C26']
data = pd.read_csv("~/dataset/criteo/small_train.txt", sep='\t', header=None, names=names)
# data = pd.read_csv("~/dataset/criteo/criteo_sampled_data.csv")
sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]
# sparse_features = ['C' + str(i) for i in range(1, 3)]
# dense_features = ['I' + str(i) for i in range(1, 3)]
data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )
target = ['label']

# 3.Label Encoding for sparse features,and do simple Transformation for dense features
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

# 4.count unique features for each sparse field,and record dense feature field name
fixlen_feature_columns = [SparseFeature(feature, vocabulary_size=data[feature].max() + 1, embedding_dim=embedding_dim)
                          for i, feature in enumerate(sparse_features)] + \
                         [DenseFeature(feature, 1, ) for feature in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 5.generate input data for model
train, test = train_test_split(data, test_size=test_size, random_state=2021)
train_model_input = {name: train[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}

# 6.define model
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = DeepFEFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=hidden_units, dnn_dropout=dropout,
                     l2_reg_dnn=l2_reg_dnn)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="binary_crossentropy",
                  metrics=['binary_crossentropy', 'AUC'])

# 7.print model summary and plot model
print(model.summary())
# tf.keras.utils.plot_model(model, to_file='./deepfefm_model.png', show_shapes=True)
# exit()
# 8.train and evaluate
history = model.fit(train_model_input, train[target].values, batch_size=batch_size, epochs=epochs,
                    validation_split=validation_size)

result = model.evaluate(test_model_input, test[target].values, batch_size=batch_size)
print(dict(zip(model.metrics_names, result)))
