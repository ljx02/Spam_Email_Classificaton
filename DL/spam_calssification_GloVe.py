# -*- coding: utf-8 -*-
# @Time  : 2020/5/22 16:53
# @Author : sjw
# @Desc : ==============================================
# If this runs wrong,don't ask me,I don't know why;  ===
# If this runs right,thank god,and I don't know why. ===
# ======================================================
# @Project : Spam_Email_Classificaton
# @FileName: spam_calssification_GloVe.py
# @Software: PyCharm
import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer  # 词干提取

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Dropout
from keras.initializers import Constant
from keras.optimizers import Adam

stemmer = PorterStemmer()
STOPWORDS = set(stopwords.words("english"))
PUNCT_TO_REMOVE = string.punctuation

train = pd.read_csv("../data/train.csv", encoding='utf-8')


def data_processing(text: str):
    text = text.lower()
    text = re.compile(r'https?://\S+|www\.\S+').sub(r'', text)
    text = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
    text = " ".join([word for word in str(text).split() if word not in STOPWORDS])
    text = " ".join([stemmer.stem(word) for word in text.split()])
    return text

def label2id(label: str):
    if label == 'ham':
        return 0
    else:
        return 1

train['Email'] = train['Email'].apply(data_processing)
train['Label'] = train['Label'].apply(label2id)
# email = np.array(train['Email']).reshape((1, len(train)))[0].tolist()
# label = np.array(train['Label']).reshape((1, len(train)))[0].tolist()

# 划分数据集，9:1
# train_email, train_label, test_email, test_label = [], [], [], []
# for i in range(len(train)):
#     if i % 10 != 0:
#         train_email.append(email[i])
#         train_label.append(label[i])
#     else:
#         test_email.append(email[i])
#         test_label.append(label[i])


def create_corpus_new(df):
    corpus = []
    for tweet in tqdm(df['Email']):
        words = [word.lower() for word in word_tokenize(tweet)]
        corpus.append(words)
    return corpus


df = train
corpus = create_corpus_new(df)

# 加载glove词向量
embedding_dict = {}
with open("../glove/glove.6B.100d.txt", 'r', encoding='utf-8')as f:
    for line in f:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:], 'float32')
        embedding_dict[word] = vectors

# 配置模型
MAX_LEN = 50
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(corpus)  # 输入为两层列表格式[[...],[...],..]
sequences = tokenizer_obj.texts_to_sequences(corpus)  # 返回序列化的数据格式
# 填充序列，大于50的会被截断，小于50的会填充补0，truncating表示截断，padding表示填充
tweet_pad = pad_sequences(sequences, maxlen=MAX_LEN, truncating='post', padding='post')

word_index = tokenizer_obj.word_index  # index从1开始
print('Number of unique words:', len(word_index))

num_words = len(word_index) + 1  # 添加0行，index从1开始
embedding_matrix = np.zeros((num_words, 100))

# 生成编码矩阵，数据集中出现过的每一个单词对应一行，对应的是词向量
for word, i in tqdm(word_index.items()):
    if i < num_words:
        emb_vec = embedding_dict.get(word)
        if emb_vec is not None:
            embedding_matrix[i] = emb_vec

# 构建模型
model = Sequential()

embedding = Embedding(num_words, 100, embeddings_initializer=Constant(embedding_matrix),
                      input_length=MAX_LEN, trainable=False)

model.add(embedding)
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(lr=3e-4)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()

train_data = tweet_pad[:3000]
test_data = tweet_pad[3000:]

X_train, X_test, y_train, y_test = train_test_split(train_data, train[:3000]['Label'].values, test_size=0.2)
print("Sahpe of train:", X_train.shape)
print("Shape of valid:", X_test.shape)

glove_model = model.fit(X_train, y_train, batch_size=4, epochs=2, validation_data=(X_test, y_test), verbose=2)

test_pred = model.predict(test_data)
test_pred_int = test_pred.round().astype('int')
loss, accuracy = model.evaluate(test_data, train[3000:]['Label'], batch_size=4, verbose=2)
print("loss:", loss)
print("accuracy: ", accuracy)
# print(test_pred_int)

