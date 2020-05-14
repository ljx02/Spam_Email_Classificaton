# -*- coding: utf-8 -*-
# @Time  : 2020/5/9 14:44
# @Author : sjw
# @Desc : ==============================================
# If this runs wrong,don't ask me,I don't know why;  ===
# If this runs right,thank god,and I don't know why. ===
# ======================================================
# @Project : text-classification
# @FileName: spam_email_classification.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import string
PUNCT_TO_REMOVE = string.punctuation

from nltk import word_tokenize
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words("english"))

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV  # 搜索适合的参数

# 读邮件数据CSV
train_email = pd.read_csv("data/train.csv", usecols=[2], encoding='utf-8')
train_label = pd.read_csv("data/train.csv", usecols=[1], encoding='utf-8')
# print(df.describe(include='all'))  # all 和 O 默认只对数字的信息进行统计
# 通过分析发现，数据中ham有3866条，总数4458，属于不平衡数据集


# 数据预处理
def text_processing(text):
    text = text.lower()
    text = re.compile(r'https?://\S+|www\.\S+').sub(r'', text)
    text = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
    text = " ".join([word for word in str(text).split() if word not in STOPWORDS])
    text = " ".join([stemmer.stem(word) for word in text.split()])
    return text


train_email['Email'] = train_email['Email'].apply(text_processing)


# 将内容转为list类型
train_email = np.array(train_email).reshape((1, len(train_email)))[0].tolist()
train_label = np.array(train_label).reshape((1, len(train_email)))[0].tolist()


# 构造训练集和验证集
train_num = int(len(train_email)*0.8)
data_train = train_email[:train_num]
data_dev = train_email[train_num:]
label_train = train_label[:train_num]
label_dev = train_label[train_num:]

# # 使用词袋模型
vectorizer = CountVectorizer()
# CountVectorizer类会把文本全部转换为小写，然后将文本词块化。主要是分词，分标点
data_train_cnt = vectorizer.fit_transform(data_train)
data_test_cnt = vectorizer.transform(data_dev)

# 第一种方法，变成TF-IDF矩阵
transformer = TfidfTransformer()
data_train_tfidf = transformer.fit_transform(data_train_cnt)
data_test_tfidf = transformer.transform(data_test_cnt)

# 第二种方法，变成TF-IDF矩阵
# vectorizer_tfidf = TfidfVectorizer(sublinear_tf=True)
# data_train_tfidf = vectorizer_tfidf.fit_transform(data_train)
# data_test_tfidf = vectorizer_tfidf.transform(data_dev)


# 利用贝叶斯的方法
clf = MultinomialNB()
clf.fit(data_train_cnt, label_train)
score = clf.score(data_test_cnt, label_dev)
print("NB score: ", score)
# 加入TF-IDF特征后打开注释
# clf.fit(data_train_tfidf, label_train)
# score = clf.score(data_test_tfidf, label_dev)
# print("NB tfidf score: ", score)

# 利用SVM的方法
svm = LinearSVC()
svm.fit(data_train_cnt, label_train)
score = svm.score(data_test_cnt, label_dev)
print("SVM score: ", score)
# 加入TF-IDF特征后打开注释
# svm.fit(data_train_tfidf, label_train)
# score = svm.score(data_test_tfidf, label_dev)
# print("SVM score: ", score)

# 利用逻辑回归的方法
lr_crf = LogisticRegression(max_iter=150, penalty='l2', solver='lbfgs', random_state=0)
lr_crf.fit(data_train_tfidf, label_train)
score = lr_crf.score(data_test_tfidf, label_dev)
print("LR score: ", score)

# 利用随机森林的方法
rf = RandomForestClassifier(random_state=0, n_estimators=100, max_depth=None, verbose=0, n_jobs=-1)
rf.fit(data_train_tfidf, label_train)
score = rf.score(data_test_tfidf, label_dev)
print("RF score: ", score)

# 利用XGBoost方法
xgb_clf = xgb.XGBClassifier(n_estimators=100, n_jobs=-1, max_depth=15, min_child_weight=3, colsample_bytree=0.4)
xgb_clf.fit(data_train_tfidf, label_train)
score = xgb_clf.score(data_test_tfidf, label_dev)
print("XGBoost score: ", score)

# 利用LightGBM的方法
lgb_clf = lgb.LGBMClassifier()
# lgb_clf.fit(data_train_tfidf, label_train)
# 使用网格搜索得到适当参数
param_test = {
    'max_depth': range(2, 3)
}
gsearch = GridSearchCV(estimator=lgb_clf, param_grid=param_test, scoring='roc_auc', cv=5)
gsearch.fit(data_train_tfidf, label_train)
# print(gsearch.best_params_)
score = gsearch.score(data_test_tfidf, label_dev)
print("LGBM score: ", score)

# 预测结果
result_lgbm = gsearch.predict(data_test_tfidf)
result_xgb = xgb_clf.predict(data_test_tfidf)
result_rf = rf.predict(data_test_tfidf)
result_lr = lr_crf.predict(data_test_tfidf)
result_svm = svm.predict(data_test_cnt)
result_nb = clf.predict(data_test_cnt)
print("NB confusion: ", confusion_matrix(label_dev, result_nb))
print("SVM confusion: ", confusion_matrix(label_dev, result_svm))
print("LR confusion: ", confusion_matrix(label_dev, result_lr))
print("RF confusion: ", confusion_matrix(label_dev, result_rf))
print("XGB confusion: ", confusion_matrix(label_dev, result_xgb))
print("LGBM confusion: ", confusion_matrix(label_dev, result_lgbm))

# 验证模型的性能
# 利用交叉验证
# accuracy = cross_val_score(clf, data_train_cnt, label_train, cv='warn', scoring='accuracy')
# print(accuracy.mean())

# 预测并写入文件，输出可以提交的格式
# 由于比赛提交需要实名认证，暂时选择不提交，后续可以选择kaggle
# test_data_id = pd.read_csv("data/test_noLabel.csv", usecols=[0], encoding='utf-8')
# test_data_email = pd.read_csv("data/test_noLabel.csv", usecols=[1], encoding='utf-8')
# test_data_email = np.array(test_data_email).reshape((1, len(test_data_email)))[0].tolist()
# test_data_id = np.array(test_data_id).tolist()
# data_test_cnt = vectorizer.transform(test_data_email)
# print(test_data_id)
# # 进行预测
# predict_label = svm.predict(data_test_cnt)
# # 写入文件
# for i, label in enumerate(predict_label):
#     test_data_id[i].append(label)
# column_name = ['ID', 'Label']
# result = pd.DataFrame(columns=column_name, data=test_data_id)
# result.to_csv("result_submit_NB.csv", index=False)
