import numpy as np
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from keras.models import Model, load_model
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.layers import (
    Input, Conv1D, AveragePooling1D, GlobalAveragePooling1D, Dropout, Dense, Activation, Concatenate, Multiply,
    BatchNormalization, Flatten, MaxPooling1D, Add, Layer, MultiHeadAttention, LayerNormalization
)
from imblearn.over_sampling import ADASYN
from sklearn.decomposition import PCA
import tcn
from keras.regularizers import l2
from keras.optimizer_v2.adam import Adam
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")
from keras import backend as K
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, multiply, Permute, Conv2D, Lambda
from collections import Counter
import itertools


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


# One-hot编码
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


# NCP编码
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


# k-mer频率特征（优化版）
def to_kmer_features(seq_list, k=3, n_components=1):
    bases = ['A', 'C', 'G', 'T']
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    kmer_tensor = np.zeros((len(seq_list), len(all_kmers)))
    for i, seq in enumerate(seq_list):
        kmers = [seq[j:j + k] for j in range(len(seq) - k + 1)]
        kmer_counts = Counter(kmers)
        for j, kmer in enumerate(all_kmers):
            kmer_tensor[i, j] = kmer_counts.get(kmer, 0) / len(kmers) if kmer in kmer_counts else 0

    # 使用PCA降维
    pca = PCA(n_components=n_components)
    kmer_reduced = pca.fit_transform(kmer_tensor)
    kmer_reduced = kmer_reduced[:, np.newaxis, :]  # 调整形状为 (样本数, 1, n_components)
    kmer_reduced = np.repeat(kmer_reduced, 41, axis=1)  # 复制到每个时间步
    return kmer_reduced


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
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 7
    channel = input_feature.shape[-1]
    cbam_feature = input_feature
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    return multiply([input_feature, cbam_feature])


# CBAM
def cbam_block(cbam_feature, ratio=8):
    channel_feature = channel_attention(cbam_feature)
    spatial_feature = spatial_attention(cbam_feature)
    X = channel_feature + spatial_feature
    return X


class CustomExpandDimsLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(CustomExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super(CustomExpandDimsLayer, self).get_config()
        config.update({'axis': self.axis})
        return config


class SqueezeLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(SqueezeLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)

    def get_config(self):
        config = super(SqueezeLayer, self).get_config()
        config.update({'axis': self.axis})
        return config


def transition(x, filters, dropout_rate, weight_decay=1e-4):
    x = Conv1D(filters=filters,
               kernel_size=1,
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = AveragePooling1D(pool_size=4)(x)
    x = BatchNormalization(axis=-1)(x)
    return x


class BiGRUModel(tf.keras.layers.Layer):
    def __init__(self, num_units, **kwargs):
        super(BiGRUModel, self).__init__(**kwargs)
        self.num_units = num_units
        self.forward_gru = tf.keras.layers.GRU(num_units, return_sequences=True)
        self.backward_gru = tf.keras.layers.GRU(num_units, return_sequences=True, go_backwards=True)
        self.concat_layer = tf.keras.layers.Concatenate()

    def call(self, inputs):
        forward_output = self.forward_gru(inputs)
        backward_output = self.backward_gru(inputs)
        return self.concat_layer([forward_output, backward_output])

    def get_config(self):
        config = super(BiGRUModel, self).get_config()
        config.update({'num_units': self.num_units})
        return config


# Focal Loss（调整参数）
def focal_loss(gamma=3.0, alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -K.mean(alpha * K.pow(1. - pt, gamma) * K.log(pt))

    return focal_loss_fixed


# 余弦退火学习率调度
def cosine_annealing(epoch, lr):
    max_lr = 5e-4
    min_lr = 1e-5
    epochs = 20
    return min_lr + (max_lr - min_lr) * (1 + np.cos(np.pi * epoch / epochs)) / 2


# 动态类权重
def compute_class_weights(labels):
    n_pos = np.sum(labels[:, 1])
    n_neg = np.sum(labels[:, 0])
    return {0: 1.0, 1: n_neg / n_pos if n_pos > 0 else 1.0}


def build_simple_model(input_shape=(41, 8), weight_decay=1e-3):  # 增强正则化
    inputs = Input(shape=input_shape)
    x = inputs

    # 优化TCN配置
    x = tcn.TCN(
        nb_filters=128,  # 减少filter数量
        kernel_size=5,  # 增大kernel size
        dilations=(1, 2, 4),  # 调整dilation rate
        dropout_rate=0.3,  # 增加dropout
        use_weight_norm=True,
        return_sequences=True
    )(x)

    # 增强正则化
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)  # 增加dropout

    # 优化BiGRU
    bigru_layer = BiGRUModel(num_units=128)  # 减少单元数
    x = bigru_layer(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)  # 新增dropout

    # 调整注意力机制
    res_out = CustomExpandDimsLayer(axis=2)(x)
    res_out_att = cbam_block(res_out, ratio=4)  # 减少压缩比例
    x = SqueezeLayer(axis=2)(res_out_att)

    # 优化全连接层
    x = Flatten()(x)
    x = Dense(128, activation="swish",  # 改用swish激活
              kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(64, activation="swish",
              kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(2, activation="softmax",
                    kernel_regularizer=l2(weight_decay))(x)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=3e-4, epsilon=1e-7)  # 调整学习率
    model.compile(loss=focal_loss(gamma=2.0, alpha=0.6),  # 优化focal参数
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

# 性能均值
def performance_mean(performance):
    print('Sn = %.4f ± %.4f' % (np.mean(performance[:, 0]), np.std(performance[:, 0])))
    print('Sp = %.4f ± %.4f' % (np.mean(performance[:, 1]), np.std(performance[:, 1])))
    print('Acc = %.4f ± %.4f' % (np.mean(performance[:, 2]), np.std(performance[:, 2])))
    print('Mcc = %.4f ± %.4f' % (np.mean(performance[:, 3]), np.std(performance[:, 3])))
    print('Auc = %.4f ± %.4f' % (np.mean(performance[:, 4]), np.std(performance[:, 4])))


if __name__ == '__main__':
    # 数据加载（保持原始数据）
    train_pos_seqs = np.array(read_fasta(r'D:\放代码的\贾博增强子\data\5mcfeiai\train_positive_data.fasta'))
    train_neg_seqs = np.array(read_fasta(r'D:\放代码的\贾博增强子\data\5mcfeiai\train_negative_data.fasta'))

    # 生成原始特征（不过采样）
    raw_train_onehot = to_one_hot(np.concatenate((train_pos_seqs, train_neg_seqs)))
    raw_train_properties = to_properties_code(np.concatenate((train_pos_seqs, train_neg_seqs)))
    raw_train_kmer = to_kmer_features(np.concatenate((train_pos_seqs, train_neg_seqs)), k=3)
    raw_train = np.concatenate((raw_train_onehot, raw_train_properties, raw_train_kmer), axis=-1)
    raw_labels = np.array([1] * len(train_pos_seqs) + [0] * len(train_neg_seqs))

    # 测试集数据（保持不变）
    test_pos_seqs = np.array(read_fasta(r'D:\放代码的\贾博增强子\data\5mcfeiai\test_positive_data.fasta'))
    test_neg_seqs = np.array(read_fasta(r'D:\放代码的\贾博增强子\data\5mcfeiai\test_negative_data.fasta'))
    test_onehot = to_one_hot(np.concatenate((test_pos_seqs, test_neg_seqs)))
    test_properties_code = to_properties_code(np.concatenate((test_pos_seqs, test_neg_seqs)))
    test_kmer = to_kmer_features(np.concatenate((test_pos_seqs, test_neg_seqs)), k=3)
    test = np.concatenate((test_onehot, test_properties_code, test_kmer), axis=-1)
    test_label = to_categorical(np.array([1] * len(test_pos_seqs) + [0] * len(test_neg_seqs)))

    # 交叉验证设置
    n = 5
    k_fold = KFold(n_splits=n, shuffle=True, random_state=42)

    # 结果存储
    sv_10_result = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    # 元模型（保持不变）
    meta_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

    # 自定义对象（添加新组件）
    custom_objects = {
        'focal_loss_fixed': focal_loss(gamma=2.0, alpha=0.6),
        'TCN': tcn.TCN,
        'BiGRUModel': BiGRUModel,
        'CustomExpandDimsLayer': CustomExpandDimsLayer,
        'SqueezeLayer': SqueezeLayer,
        'cbam_block': cbam_block
    }

    for k in range(2):
        print('\n' + '*' * 30 + f' 第 {k} 轮 ' + '*' * 30)
        fold_metrics = []
        val_pred_all = []
        test_pred_all = []
        val_labels_all = []

        for fold_idx, (train_idx, val_idx) in enumerate(k_fold.split(raw_train)):
            print('\n' + '*' * 30 + f' 折 {fold_idx + 1} ' + '*' * 30)

            # 正确应用 SMOTE（仅在训练集）
            tra_flat = raw_train[train_idx].reshape(len(train_idx), -1)
            tra_label_flat = raw_labels[train_idx]

            # 控制过采样比例
            tra_resampled, tra_label_resampled = SMOTE(sampling_strategy=0.8, random_state=42).fit_resample(
                tra_flat, tra_label_flat
            )

            tra = tra_resampled.reshape(-1, 41, 8)
            tra_label = to_categorical(tra_label_resampled)

            # 验证集保持原始分布
            val = raw_train[val_idx]
            val_label = to_categorical(raw_labels[val_idx])

            # 模型构建
            model = build_simple_model()

            # 训练参数优化
            BATCH_SIZE = 256
            EPOCHS = 30


            # 学习率调度
            def dynamic_lr(epoch):
                if epoch < 10:
                    return 3e-4
                elif epoch < 20:
                    return 1e-4
                else:
                    return 5e-5


            # 早停策略
            early_stop = EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                mode='max',
                restore_best_weights=True
            )

            # 模型训练
            history = model.fit(
                x=tra,
                y=tra_label,
                validation_data=(val, val_label),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                callbacks=[early_stop, LearningRateScheduler(dynamic_lr)],
                verbose=1
            )

            # 阈值优化函数
            from sklearn.metrics import matthews_corrcoef


            def optimize_threshold(y_true, y_pred):
                thresholds = np.linspace(0.3, 0.7, 100)
                mcc_values = [matthews_corrcoef(y_true, (y_pred >= t).astype(int)) for t in thresholds]
                return thresholds[np.argmax(mcc_values)]


            # 验证集预测
            val_pred = model.predict(val, verbose=0)[:, 1]
            best_thresh = optimize_threshold(val_label[:, 1], val_pred)

            # 测试集预测
            test_pred = model.predict(test, verbose=0)[:, 1]
            test_pred_bin = (test_pred >= best_thresh).astype(int)

            # 性能计算
            Sn, Sp, Acc, MCC = show_performance(test_label[:, 1], test_pred_bin)
            AUC = roc_auc_score(test_label[:, 1], test_pred)
            print(f'Threshold={best_thresh:.2f}, Sn={Sn:.4f}, Sp={Sp:.4f}, Acc={Acc:.4f}, MCC={MCC:.4f}, AUC={AUC:.4f}')

            # 存储结果
            val_pred_all.append(val_pred)
            test_pred_all.append(test_pred)
            val_labels_all.append(val_label[:, 1])
            fold_metrics.append([Sn, Sp, Acc, MCC, AUC])

            # 模型保存
            model.save(f'../models/5mc_model_{k}_fold{fold_idx}.h5')
            tf.keras.backend.clear_session()

        # 元模型训练
        meta_model.fit(np.array(val_pred_all).T, np.concatenate(val_labels_all))

        # 集成预测
        stacking_pred = meta_model.predict_proba(np.array(test_pred_all).T)[:, 1]
        final_pred = (stacking_pred >= optimize_threshold(test_label[:, 1], stacking_pred)).astype(int)

        # 最终评估
        Sn, Sp, Acc, MCC = show_performance(test_label[:, 1], final_pred)
        AUC = roc_auc_score(test_label[:, 1], stacking_pred)
        sv_10_result.append([Sn, Sp, Acc, MCC, AUC])

        # 绘制 ROC 曲线
        fpr, tpr, _ = roc_curve(test_label[:, 1], stacking_pred)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        plt.plot(fpr, tpr, label=f'Round {k} (AUC={AUC:.2f})')

    # 最终输出
    print('\n' + '=' * 40 + ' Final Performance ' + '=' * 40)
    performance_mean(np.array(sv_10_result))

    # 绘制平均 ROC 曲线
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean([x[4] for x in sv_10_result])
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC={mean_auc:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig('../results/optimized_roc_curve.jpg', dpi=300)
    plt.show()
