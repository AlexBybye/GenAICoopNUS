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
    output_sequence_length=max_length
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
embedding_dim = 96  # 减少嵌入维度 (128→96)
dropout_rate = 0.65 # 增加Dropout强度 (0.6→0.65)
learning_rate = 0.0002  # 降低学习率 (0.0003→0.0002)

# ===== 改进的模型架构 =====
model = Sequential([
    Embedding(vocab_size, embedding_dim),  # 移除input_length参数
    Dropout(dropout_rate),
    GlobalAveragePooling1D(),
    Dense(48, activation='relu', kernel_regularizer=regularizers.l2(0.001)),  # 减少神经元+添加L2正则化
    Dropout(dropout_rate),
    Dense(24, activation='relu'),  # 减少神经元
    Dropout(dropout_rate/2),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=learning_rate),
    metrics=['accuracy']
)

# ===== 增强的早停和学习率衰减 =====
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

# 训练模型
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[
        keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True),
        early_stopping,
        lr_scheduler
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

plot_graphs(history, "accuracy", "Optimized_Model_Accuracy.png")
plot_graphs(history, "loss", "Optimized_Model_Loss.png")

# ===== 测试集评估 =====
print("\n" + "="*50)
print("在测试集上评估最佳模型")
print("="*50)

best_model = keras.models.load_model('best_model.keras')
test_loss, test_acc = best_model.evaluate(X_test, y_test)

print(f"\n测试集评估结果:")
print(f"- 损失: {test_loss:.4f}")
print(f"- 准确率: {test_acc:.4f}")