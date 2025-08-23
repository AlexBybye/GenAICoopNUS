import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow_datasets as tfds
import numpy as np
import re, string
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers


# ===== 新增：自定义早停回调 =====
class ValAccImprovementStopper(keras.callbacks.Callback):
    def __init__(self, min_improvement=0.005, patience=15, verbose=1):
        """
        当验证准确率提升小于min_improvement持续patience个epoch时停止训练
        min_improvement: 最小提升阈值 (0.005 = 0.5%)
        patience: 连续满足条件的epoch数
        """
        super(ValAccImprovementStopper, self).__init__()
        self.min_improvement = min_improvement
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.best_val_acc = 0.0
        self.best_weights = None
        self.best_epoch = 0

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best_val_acc = 0.0
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current_val_acc = logs.get('val_accuracy')

        # 更新最佳验证准确率
        if current_val_acc > self.best_val_acc:
            improvement = current_val_acc - self.best_val_acc
            self.best_val_acc = current_val_acc
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch
            self.wait = 0  # 重置计数器

            if self.verbose > 0:
                print(f"\n验证准确率提升: +{improvement:.4f} (新最佳: {self.best_val_acc:.4f})")
        else:
            # 计算与最佳准确率的差距
            improvement = current_val_acc - self.best_val_acc

            # 检查提升是否小于阈值
            if improvement < self.min_improvement:
                self.wait += 1
                if self.verbose > 0:
                    print(f"\n验证准确率提升不足: +{improvement:.4f} < {self.min_improvement:.4f} "
                          f"(第{self.wait}/{self.patience}轮)")
            else:
                self.wait = 0

        # 满足停止条件
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.best_weights is not None:
                self.model.set_weights(self.best_weights)
            if self.verbose > 0:
                print(f"\n早停触发: 连续{self.patience}轮验证准确率提升小于{self.min_improvement:.4f}")
                print(f"最佳验证准确率: {self.best_val_acc:.4f} (epoch {self.best_epoch})")


# Stopwords list (保持不变)
stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "hed", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how",
             "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "it", "its", "itself",
             "lets", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "shed", "shell", "shes", "should",
             "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then",
             "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "wed", "well", "were", "weve", "were",
             "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why",
             "whys", "with", "would", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself",
             "yourselves"]


# 标准化函数 (保持不变)
def custom_standardization(input_data):
    data = tf.strings.lower(input_data)
    data = tf.strings.regex_replace(data, f"[{re.escape(string.punctuation)}]", "")
    data = tf.strings.regex_replace(data, '<br />', ' ')
    pattern = r'\b(' + '|'.join(stopwords) + r')\b'
    data = tf.strings.regex_replace(data, pattern, ' ')
    data = tf.strings.regex_replace(data, r'\s+', ' ')
    return data


# Tokenizer split function (保持不变)
def custom_split(input_data):
    return tf.strings.split(input_data)


# Parameters (保持不变)
max_length = 600
vocab_size = 20000

# Text vectorization layer (保持不变)
text_vectorization = TextVectorization(
    output_mode='int',
    standardize=custom_standardization,
    split=custom_split,
    max_tokens=vocab_size,
    output_sequence_length=max_length,
    pad_to_max_tokens=True
)

# 数据加载和预处理 (保持不变)
train_data_raw = tfds.as_numpy(tfds.load('imdb_reviews', split="train"))
test_data_raw = tfds.as_numpy(tfds.load('imdb_reviews', split="test"))
train_texts = [str(item['text']) for item in train_data_raw]
train_labels = [int(item['label']) for item in train_data_raw]
test_texts = [str(item['text']) for item in test_data_raw]
test_labels = [int(item['label']) for item in test_data_raw]
text_vectorization.adapt(train_texts)

# 训练验证分割 (保持不变)
from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42
)
X_train = text_vectorization(np.array(train_texts))
X_val = text_vectorization(np.array(val_texts))
X_test = text_vectorization(np.array(test_texts))
y_train = np.array(train_labels)
y_val = np.array(val_labels)
y_test = np.array(test_labels)

# ===== 参数优化 =====
embedding_dim = 256  # 增加嵌入维度 (原144→256) - 提供更丰富的词向量表示
dropout_rate = 0.6  # 微调Dropout率 (原0.52→0.55) - 平衡正则化强度
learning_rate = 0.00023  # 学习率微调 (原0.00015→0.0002) - 加速前期收敛
batch_size = 32  # 调整批量大小 (原24→32) - 提高训练效率

# ===== 改进的模型架构 =====
# 应用新的正则化方法：增加中间层维度并应用L2正则化
new_layer_dim = 128  # 新的中间层维度 - 提供更强的表示能力
l2_lambda = 0.001  # L2正则化系数 - 控制正则化强度

model = Sequential([
    Embedding(vocab_size, embedding_dim,mask_zero=True),  # 移除input_length参数
    Dropout(dropout_rate),
    GlobalAveragePooling1D(),

    # 新增的中间层，应用L2正则化 - 这是核心改动
    Dense(new_layer_dim, activation='relu',
          kernel_regularizer=regularizers.l2(l2_lambda),
          bias_regularizer=regularizers.l2(0.0005)),

    Dropout(dropout_rate),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=learning_rate),
    metrics=['accuracy']
)

# ===== 增强的早停和学习率衰减 =====
# 调整回调参数以配合新架构
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=8,  # 增加耐心值 (原5→8) - 给新架构更多收敛时间
    restore_best_weights=True
)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,  # 增加耐心值 (原2→3) - 避免过早降低学习率
    min_lr=1e-7,
    verbose=1  # 显示学习率变化
)

# ===== 创建自定义回调实例 =====
val_acc_stopper = ValAccImprovementStopper(
    min_improvement=0.005,  # 0.5%的提升阈值
    patience=15,  # 连续15轮
    verbose=1
)

# 训练模型 - 增加epoch以充分利用新架构
history = model.fit(
    X_train, y_train,
    epochs=60,  # 增加训练轮次 (原50→80)
    batch_size=batch_size,
    validation_data=(X_val, y_val),
    callbacks=[
        keras.callbacks.ModelCheckpoint('best_model.keras',
                                        monitor='val_accuracy',  # 监控验证准确率
                                        save_best_only=True,
                                        mode='max'),
        early_stopping,
        lr_scheduler,
        val_acc_stopper  # 添加自定义回调
    ],
    verbose=2
)


# 可视化函数 (保持不变)
def plot_graphs(history, metric, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history[metric], label='Train')
    plt.plot(history.history['val_' + metric], label='Validation')
    best_epoch = np.argmin(history.history['val_loss'])
    best_value = history.history['val_' + metric][best_epoch]
    plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
    plt.scatter(best_epoch, best_value, color='red', s=50, zorder=5)
    plt.xlabel("Epochs")
    plt.ylabel(metric.capitalize())
    plt.title(f"{metric.capitalize()} Over Epochs (Best: {best_value:.4f})")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


plot_graphs(history, "accuracy", "Regularization_Model_Accuracy.png")
plot_graphs(history, "loss", "Regularization_Model_Loss.png")

# ===== 测试集评估 =====
print("\n" + "=" * 50)
print("在测试集上评估最佳模型")
print("=" * 50)

# 优先使用ModelCheckpoint保存的最佳模型
try:
    best_model = keras.models.load_model('best_model.keras')
    print("使用ModelCheckpoint保存的最佳模型")
except:
    # 如果文件不存在，使用自定义回调保存的最佳权重
    model.set_weights(val_acc_stopper.best_weights)
    best_model = model
    print("使用自定义回调保存的最佳模型")

test_loss, test_acc = best_model.evaluate(X_test, y_test)

print(f"\n测试集评估结果:")
print(f"- 损失: {test_loss:.4f}")
print(f"- 准确率: {test_acc:.4f}")

# 显示最佳验证准确率信息
if hasattr(val_acc_stopper, 'best_val_acc'):
    print(f"- 最佳验证准确率: {val_acc_stopper.best_val_acc:.4f} (epoch {val_acc_stopper.best_epoch})")