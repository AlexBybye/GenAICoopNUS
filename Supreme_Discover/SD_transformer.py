import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

import re
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

plt.style.use('default')
plt.rcParams["figure.figsize"] = (12, 8)


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weight", shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.matmul(tf.reshape(x, [-1, x.shape[-1]]), self.W)
        e = tf.reshape(e, [-1, x.shape[1]])
        alpha = tf.nn.softmax(e)
        output = x * tf.expand_dims(alpha, -1)
        output = tf.reduce_sum(output, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


# GPU配置
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU enabled: {len(gpus)} GPU(s) available")
    except RuntimeError as e:
        print(f"GPU config error: {e}")



from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))

class IMDBSentimentAnalyzer:
    def __init__(self, **params):
        # 优化参数配置，目标92%精度
        self.default_params = {
            'vocab_size': 20000,  # 增大词汇表
            'max_length': 600,  # 序列长度
            'embedding_dim': 128,  # 增大嵌入维度
            'lstm_units': 200,  # 增大LSTM单元
            'dropout_rate': 0.7,  # 降低dropout
            'recurrent_dropout': 0.2,
            'learning_rate': 0.0001,  # 降低学习率
            'batch_size': 64,  # 减小批次大小
            'use_attention': True,
            'use_batch_norm': True,
            'l2_reg': 0.0001,  # 减小正则化
            'dense_units': 128
        }

        self.params = {**self.default_params, **params}
        self.tokenizer = None
        self.model = None
        self.history = None

        # 基础停用词
        self.stopwords = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
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
            "yourselves"
        }
        print("IMDB Sentiment Analyzer initialized")

    def load_imdb_data(self):
        print("Loading IMDB dataset...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
            num_words=self.params['vocab_size']
        )

        word_index = tf.keras.datasets.imdb.get_word_index()
        reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

        def decode_review(text):
            return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

        train_texts = [decode_review(review) for review in x_train]
        test_texts = [decode_review(review) for review in x_test]

        print(f"Training samples: {len(train_texts)}")
        print(f"Test samples: {len(test_texts)}")

        self.raw_train_texts = train_texts
        self.raw_train_labels = y_train
        self.raw_test_texts = test_texts
        self.raw_test_labels = y_test

        self._visualize_data_distribution()

        return (train_texts, y_train), (test_texts, y_test)

    def _visualize_data_distribution(self):
        train_lengths = [len(text.split()) for text in self.raw_train_texts[:5000]]

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.hist(train_lengths, bins=50, alpha=0.7, color='skyblue')
        plt.axvline(np.mean(train_lengths), color='red', linestyle='--',
                    label=f'Mean: {np.mean(train_lengths):.1f}')
        plt.axvline(self.params['max_length'], color='green', linestyle='-',
                    label=f'Max length: {self.params["max_length"]}')
        plt.title('Review Length Distribution')
        plt.xlabel('Number of words')
        plt.ylabel('Number of reviews')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        train_pos = np.sum(self.raw_train_labels)
        train_neg = len(self.raw_train_labels) - train_pos
        test_pos = np.sum(self.raw_test_labels)
        test_neg = len(self.raw_test_labels) - test_pos

        x = np.arange(2)
        width = 0.35
        plt.bar(x - width / 2, [train_pos, train_neg], width, label='Training', alpha=0.7)
        plt.bar(x + width / 2, [test_pos, test_neg], width, label='Test', alpha=0.7)

        plt.title('Label Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of reviews')
        plt.xticks(x, ['Positive', 'Negative'])
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('data_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

    def clean_text(self, text):
        text = str(text).lower()
        # 处理HTML标签
        text = re.sub(r"<br\s*/?>", " ", text)
        text = re.sub(r"&[a-z]+;", " ", text)

        # 处理缩写
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r"'m", " am", text)
        text = re.sub(r"'s", " is", text)

        # 保留情感符号
        text = re.sub(r"(!+)", r" \1 ", text)
        text = re.sub(r"(\?+)", r" \1 ", text)

        # 处理重复字符
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        # 保留字母数字和重要标点
        text = re.sub(r'[^a-zA-Z0-9\s!?.]', ' ', text)

        # 分词和过滤
        words = text.split()
        filtered_words = [word for word in words
                          if (len(word) > 2 or word in ['!', '?']) and
                          word not in self.stopwords]

        if not filtered_words:
            return "empty review"
        return ' '.join(filtered_words)

    def preprocess_data(self, texts, labels=None, is_training=True):
        print(f"Cleaning {len(texts)} reviews...")
        cleaned_texts = [self.clean_text(text) for text in texts]

        if is_training and self.tokenizer is None:
            print("Creating tokenizer...")
            self.tokenizer = Tokenizer(
                num_words=self.params['vocab_size'],
                oov_token='<OOV>',
                lower=True
            )
            self.tokenizer.fit_on_texts(cleaned_texts)
            print(f"Vocabulary size: {len(self.tokenizer.word_index)}")

        print("Converting to sequences...")
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)

        print("Padding sequences...")
        padded = pad_sequences(
            sequences,
            padding='post',
            maxlen=self.params['max_length'],
            truncating='post'
        )
        print(f"Preprocessing complete, data shape: {padded.shape}")

        if labels is not None:
            return padded, np.array(labels)
        return padded

    def build_model(self):
        print("Building Transformer Encoder model...")
        inputs = Input(shape=(self.params['max_length'],))

        x = Embedding(
            input_dim=self.params['vocab_size'],
            output_dim=self.params['embedding_dim'],
            input_length=self.params['max_length']
        )(inputs)

        # 减少Transformer块为1个
        x = TransformerBlock(
            embed_dim=self.params['embedding_dim'],
            num_heads=4,
            ff_dim=64,  # 减小FFN维度
            rate=self.params['dropout_rate']
        )(x)

        x = GlobalAveragePooling1D()(x)
        if self.params['use_batch_norm']:
            x = BatchNormalization()(x)

        x = Dense(self.params['dense_units'], activation='relu',
                  kernel_regularizer=l2(self.params['l2_reg']))(x)
        x = Dropout(self.params['dropout_rate'])(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=self.params['learning_rate'])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        self.model = model
        print(f"Model parameters: {model.count_params():,}")
        return model

    def train(self, texts, labels, validation_split=0.15, epochs=30, verbose=1):
        X, y = self.preprocess_data(texts, labels, is_training=True)

        # 回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                mode='max',
                patience=5,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=verbose
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=verbose
            )
        ]

        print("Training model...")
        self.history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=self.params['batch_size'],
            callbacks=callbacks,
            verbose=verbose
        )

        self._plot_training_history()
        return self.history

    def _plot_training_history(self):
        if self.history is None:
            return

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

    def evaluate(self, texts, labels, detailed=True):
        X, y = self.preprocess_data(texts, labels, is_training=False)
        predictions = self.model.predict(X, verbose=0)
        y_pred = (predictions > 0.5).astype(int).flatten()

        accuracy = accuracy_score(y, y_pred)

        if detailed:
            print(f"\nTest Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y, y_pred, target_names=['Negative', 'Positive']))

        # 可视化结果
        self._visualize_evaluation_results(y, y_pred, predictions)

        return {'accuracy': accuracy}

    def _visualize_evaluation_results(self, y_true, y_pred, predictions):
        from sklearn.metrics import confusion_matrix

        plt.figure(figsize=(12, 4))

        # 混淆矩阵
        plt.subplot(1, 3, 1)
        cm = confusion_matrix(y_true, y_pred)
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Negative', 'Positive'])
        plt.yticks(tick_marks, ['Negative', 'Positive'])

        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2. else "black")

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # 预测概率分布
        plt.subplot(1, 3, 2)
        plt.hist(predictions[y_true == 1], bins=30, alpha=0.6, label='Positive', color='green')
        plt.hist(predictions[y_true == 0], bins=30, alpha=0.6, label='Negative', color='red')
        plt.axvline(0.5, color='black', linestyle='--', label='Threshold')
        plt.title('Prediction Probability Distribution')
        plt.xlabel('Positive Probability')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 置信度分析
        plt.subplot(1, 3, 3)
        confidence = np.maximum(predictions, 1 - predictions)
        correct = (y_pred == y_true)
        plt.hist(confidence[correct], bins=30, alpha=0.6, label='Correct', color='blue')
        plt.hist(confidence[~correct], bins=30, alpha=0.6, label='Incorrect', color='orange')
        plt.title('Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

    def predict_sentiment(self, texts, return_proba=False):
        X = self.preprocess_data(texts, is_training=False)
        predictions = self.model.predict(X, verbose=0)
        if return_proba:
            return predictions.flatten()
        else:
            return (predictions > 0.5).astype(int).flatten()


# 主程序
def main():
    # 使用优化参数
    analyzer = IMDBSentimentAnalyzer()

    # 加载数据
    (train_texts, train_labels), (test_texts, test_labels) = analyzer.load_imdb_data()

    # 构建和训练模型
    analyzer.build_model()

    # 使用完整数据集训练
    analyzer.train(
        train_texts,
        train_labels,
        validation_split=0.15,
        epochs=35,  # 增加训练轮次
        verbose=1
    )

    # 评估模型
    metrics = analyzer.evaluate(test_texts, test_labels)

    print(f"\nFinal Test Accuracy: {metrics['accuracy']:.4f}")
    # 示例预测
    sample_reviews = [
        "This movie is absolutely fantastic! Great acting and amazing plot.",
        "Terrible film. Boring and poorly made. Complete waste of time.",
        "An okay movie. Nothing special but watchable."
    ]

    print("\nSample Predictions:")
    predictions = analyzer.predict_sentiment(sample_reviews, return_proba=True)
    for i, (review, prob) in enumerate(zip(sample_reviews, predictions)):
        sentiment = "Positive" if prob > 0.5 else "Negative"
        confidence = prob if prob > 0.5 else 1 - prob
        print(f"{i + 1}. {sentiment} ({confidence:.3f}): {review[:60]}...")

    return analyzer, metrics


if __name__ == "__main__":
    tf.keras.backend.clear_session()
    analyzer, metrics = main()
