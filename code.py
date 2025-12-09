import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from keras.models import Model, load_model
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import (
    Input, Conv1D, AveragePooling1D, GlobalAveragePooling1D, Dropout, Dense, Activation, Concatenate, Multiply,
    BatchNormalization, Flatten, MaxPooling1D,Add,Layer)
import tcn
from tensorflow.keras import layers, models
from tensorflow.keras import layers
from keras.regularizers import l2
from keras.optimizer_v2.adam import Adam
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.regularizers import l2,l1
warnings.filterwarnings("ignore")
from tensorflow.keras.layers import MultiHeadAttention, Input, Dense, Dropout, BatchNormalization, Flatten, Concatenate, Conv1D, MaxPooling1D, Bidirectional, LSTM
from keras import backend as K
from keras import initializers,layers,regularizers
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
# 读取DNA序列
def read_fasta(fasta_file_name):
    seqs = []
    seqs_num = 0
    file = open(fasta_file_name)
    for line in file.readlines():
        if line.strip() == '':
            continue
        if line.startswith('>'):
            seqs_num = seqs_num + 1
            continue
        else:
            seq = line.strip()

            result1 = 'N' in seq
            result2 = 'n' in seq
            if result1 == False and result2 == False:
                seqs.append(seq)
    return seqs


# One-hot coding
def to_one_hot(seq_list):
    tensor = np.zeros((len(seq_list), 41, 4))
    for i in range(len(seq_list)):
        seq = seq_list[i]
        j = 0
        for s in seq:
            if s == 'A':
                tensor[i][j] = [1, 0, 0, 0]
            if s == 'T':
                tensor[i][j] = [0, 1, 0, 0]
            if s == 'C':
                tensor[i][j] = [0, 0, 1, 0]
            if s == 'G':
                tensor[i][j] = [0, 0, 0, 1]
            j += 1
    return tensor

# NCP coding
def to_properties_code(seq_list):
    tensor = np.zeros((len(seq_list), 41, 3))
    for i in range(len(seq_list)):
        seq = seq_list[i]
        j = 0
        for s in seq:
            if s == 'A':
                tensor[i][j] = [1, 1, 1]
            if s == 'T':
                tensor[i][j] = [0, 0, 1]
            if s == 'C':
                tensor[i][j] = [0, 1, 0]
            if s == 'G':
                tensor[i][j] = [1, 0, 0]
            j += 1
    return tensor



# 性能评估
def show_performance(y_true, y_pred):
    TP = FP = FN = TN = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred > 0.5:
            TP += 1
        elif true == 1:
            FN += 1
        elif pred > 0.5:
            FP += 1
        else:
            TN += 1
    Sn = TP / (TP + FN + 1e-6)
    Sp = TN / (FP + TN + 1e-6)
    Acc = (TP + TN) / len(y_true)
    MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-6)
    return Sn, Sp, Acc, MCC



# Channel Attention Module
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(
        input_feature)  # GlobalAveragePooling2D 全局平均池化  只剩下batchsize与channel两个维度。从形状上看：[B,H,W,C] → [B,C]
    avg_pool = Reshape((1, 1, channel))(avg_pool)  # 改变shape：宽度，高度，深度（拉成一个向量，这样才能喂到MLP）
    # assert avg_pool._keras_shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    # assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    # assert avg_pool._keras_shape[1:] == (1,1,channel)

    max_pool = GlobalMaxPooling2D()(
        input_feature)  # GlobalMaxPooling2D 全局最大池化  只剩下batchsize与channel两个维度。从形状上看：[B,H,W,C] → [B,C]
    max_pool = Reshape((1, 1, channel))(max_pool)
    # assert max_pool._keras_shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    # assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    # assert max_pool._keras_shape[1:] == (1,1,channel)

    cbam_feature = Add()([avg_pool, max_pool])  # 处理后的结果相加

    cbam_feature = Activation('sigmoid')(cbam_feature)  # 获得各通道的权重图

    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7

    channel = input_feature.shape[-1]
    cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)  # 对张量求平均值，改变第三维坐标，并保持原本维度
    # assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    # assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])  # 拼接
    # assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    #对拼接后的结果进行 2D 卷积操作，输出一个通道数为 1 的特征图，并使用 sigmoid 激活函数进行激活
    return multiply([input_feature, cbam_feature])


# CBAM
def cbam_block(cbam_feature, ratio=8):
    channel_feature = channel_attention(cbam_feature)#调用之前的函数
    spatial_feature = spatial_attention(cbam_feature)
    X = channel_feature+spatial_feature
    return X


class CustomExpandDimsLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(CustomExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)


class SqueezeLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(SqueezeLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)


def transition(x, filters, dropout_rate, weight_decay=1e-4):
    # x = Activation('relu')(x)
    x = Conv1D(filters=filters,
               kernel_size=1,
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(dropout_rate)(x)
    x = AveragePooling1D(pool_size=4)(x)
    x = BatchNormalization(axis=-1)(x)
    return x


class BiGRUModel(tf.keras.Model):
    def __init__(self, num_units, **kwargs):
        super(BiGRUModel, self).__init__(**kwargs)
        self.forward_gru = tf.keras.layers.GRU(num_units, return_sequences=True)
        self.backward_gru = tf.keras.layers.GRU(num_units, return_sequences=True, go_backwards=True)
        self.concat_layer = tf.keras.layers.Concatenate()

    def call(self, inputs):
        forward_output = self.forward_gru(inputs)
        backward_output = self.backward_gru(inputs)
        combined_output = self.concat_layer([forward_output, backward_output])
        return combined_output



def focal_loss(gamma=2):
    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred)
        y_true = tf.cast(y_true, dtype=tf.int32)
        ce_loss = tf.keras.losses.sparse_categorical_cross_entropy(y_true, y_pred)
        pt = tf.gather(y_pred, y_true, batch_dims=1)
        focal_loss_value = -((1 - pt) ** gamma * tf.math.log(pt))
        return tf.reduce_mean(focal_loss_value)

    return focal_loss_fn





def build_simple_model(input_shape=(41, 7), num_classes=2,dilation_rates=[1, 2, 4, 8],
                       weight_decay=1e-4,tcn_filters=[128],tcn_filters1=128, kernel_size=3,dropout_rate=0.5):
    inputs = Input(shape=input_shape)
    x = inputs
    x = tcn.TCN(nb_filters=256, kernel_size=3, nb_stacks=2, dilations=(1, 2, 4, 8), return_sequences=True,
                 go_backwards=False, use_weight_norm=True)(x)
    x = transition(x, filters=tcn_filters1, dropout_rate=dropout_rate, weight_decay=weight_decay)
    model = BiGRUModel(num_units=256)
    x = model(x)
    #  添加注意力机制
    res_out = CustomExpandDimsLayer(axis=2)(x)
    res_out_att = cbam_block(res_out, ratio=8)
    x = SqueezeLayer(axis=2)(res_out_att)
    x = Flatten()(x)

    # MLP
    x = Dense(units=256, activation="sigmoid", use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(units=128, activation="sigmoid", use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.3)(x)
    x = Dense(units=2, activation="softmax", use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x)
    model = Model(inputs=inputs, outputs=x, name="simple_enhancer")
    optimizer = Adam(lr=5e-4, epsilon=5e-8)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model



# 性能均值
def performance_mean(performance):
    print('Sn = %.4f ± %.4f' % (np.mean(performance[:, 0]), np.std(performance[:, 0])))
    print('Sp = %.4f ± %.4f' % (np.mean(performance[:, 1]), np.std(performance[:, 1])))
    print('Acc = %.4f ± %.4f' % (np.mean(performance[:, 2]), np.std(performance[:, 2])))
    print('Mcc = %.4f ± %.4f' % (np.mean(performance[:, 3]), np.std(performance[:, 3])))
    print('Auc = %.4f ± %.4f' % (np.mean(performance[:, 4]), np.std(performance[:, 4])))


if __name__ == '__main__':
    # 读取训练集
    train_pos_seqs = np.array(read_fasta(r''))
    train_neg_seqs = np.array(read_fasta(r''))
    train_seqs = np.concatenate((train_pos_seqs, train_neg_seqs), axis=0)


    train_onehot = np.array(to_one_hot(train_seqs)).astype(np.float32)
    train_properties_code = np.array(to_properties_code(train_seqs)).astype(np.float32)
    # train_nd_code = np.array(nd_encoding(train_seqs)).astype(np.float32)
    # train_phy_code = np.array(physical_structural(train_seqs, transposed_phy_value_scale)).astype(np.float32)

    train = np.concatenate((train_onehot, train_properties_code), axis=-1)
    # train2 = np.concatenate((train_nd_code, train_phy_code), axis=-1)
    train_label = np.array([1] * 55800 + [0] * 658858).astype(np.float32)
    train_label = to_categorical(train_label, num_classes=2)
    # train = np.concatenate((train1, train2), axis=-1)

    # 读取测试集
    test_pos_seqs = np.array(read_fasta(r''))
    test_neg_seqs = np.array(read_fasta(r''))
    test_seqs = np.concatenate((test_pos_seqs, test_neg_seqs), axis=0)

    test_onehot = np.array(to_one_hot(test_seqs)).astype(np.float32)
    test_properties_code = np.array(to_properties_code(test_seqs)).astype(np.float32)
    # test_nd_code = np.array(nd_encoding(test_seqs)).astype(np.float32)
    # test_phy_code = np.array(physical_structural(test_seqs, transposed_phy_value_scale)).astype(np.float32)

    test = np.concatenate((test_onehot, test_properties_code), axis=-1)
    # test2 = np.concatenate((test_nd_code, test_phy_code), axis=-1)
    test_label = np.array([1] * 13950 + [0] * 164713).astype(np.float32)
    test_label = to_categorical(test_label, num_classes=2)
    # test = np.concatenate((test1, test2), axis=-1)

    # 交叉验证
    n = 5
    k_fold = KFold(n_splits=n, shuffle=True, random_state=1337)

    sv_10_result = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    for k in range(20):#循环
        print('*' * 30 + ' the ' + str(k) + ' cycle ' + '*' * 30)
        all_Sn = []
        all_Sp = []
        all_Acc = []
        all_MCC = []
        all_AUC = []
        test_pred_all = []
        mean_fpr = np.linspace(0, 1, 100)
        for fold_count, (train_index, val_index) in enumerate(k_fold.split(train)):
            print('*' * 30 + ' fold ' + str(fold_count + 1) + ' ' + '*' * 30)
            tra, val = train[train_index], train[val_index]
            tra_label, val_label = train_label[train_index], train_label[val_index]

            model = build_simple_model()

            BATCH_SIZE = 1024
            EPOCHS = 10

            history = model.fit(x=tra, y=tra_label, validation_data=(val, val_label), epochs=EPOCHS,
                                batch_size=BATCH_SIZE, shuffle=True,
                                callbacks=[EarlyStopping(monitor='val_loss', patience=40, mode='auto')],
                                verbose=1)


            train_loss = history.history["loss"]
            train_acc = history.history["accuracy"]
            val_loss = history.history["val_loss"]
            val_acc = history.history["val_accuracy"]

            loss, accuracy = model.evaluate(val, val_label, verbose=1)

            print('val loss:', loss)
            print('val accuracy:', accuracy)

            model.save('../models/5mc1_model_' + str(fold_count))

            del model

            model = load_model('../models/5mc1_model_' + str(fold_count))

            test_pred = model.predict(test, verbose=1)
            test_pred_all.append(test_pred[:, 1])

            Sn, Sp, Acc, MCC = show_performance(test_label[:, 1], test_pred[:, 1])
            AUC = roc_auc_score(test_label[:, 1], test_pred[:, 1])
            print('Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f' % (Sn, Sp, Acc, MCC, AUC))

            all_Sn.append(Sn)
            all_Sp.append(Sp)
            all_Acc.append(Acc)
            all_MCC.append(MCC)
            all_AUC.append(AUC)
        fold_count += 1
        fold_avg_Sn = np.mean(all_Sn)
        fold_avg_Sp = np.mean(all_Sp)
        fold_avg_Acc = np.mean(all_Acc)
        fold_avg_MCC = np.mean(all_MCC)
        fold_avg_AUC = np.mean(all_AUC)

        test_pred_all = np.array(test_pred_all).T
        ruan_voting_test_pred = test_pred_all.mean(axis=1)

        sv_Sn, sv_Sp, sv_Acc, sv_MCC = show_performance(test_label[:, 1], ruan_voting_test_pred)
        sv_AUC = roc_auc_score(test_label[:, 1], ruan_voting_test_pred)
        sv_result = [sv_Sn, sv_Sp, sv_Acc, sv_MCC, sv_AUC]
        sv_10_result.append(sv_result)

        fpr, tpr, thresholds = roc_curve(test_label[:, 1], ruan_voting_test_pred, pos_label=1)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        plt.plot(fpr, tpr, label='ROC cycle {} (AUC={:.4f})'.format(str(k), sv_AUC))

    print('---------------------------------------------soft voting 10---------------------------------------')
    print(np.array(sv_10_result))
    performance_mean(np.array(sv_10_result))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.plot([0, 1], [0, 1], '--', color='red')
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(np.array(sv_10_result)[:, 4])

    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC=%0.4f)' % (mean_auc), lw=2, alpha=.8)
    plt.title('   ')
    plt.savefig('../images/ROC_Curve_of_5mc.jpg', dpi=1200, bbox_inches='tight')
    plt.legend(loc='lower right')
    plt.show()
