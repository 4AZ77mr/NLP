import os
import sys

sys.path.append("libsvm/python/")

from svm import svm_parameter, svm_problem
import svmutil

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from preprocessing import MecabTokenizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from gensim import corpora, matutils

from multiprocessing import Pool

from collections import Counter

import pandas as pd
import numpy as np

import time

pre_sum = 0
rec_sum = 0
f1_sum = 0
modelselection = {
                    'K-NN'  : KNeighborsClassifier(n_neighbors=1),
                    'NB'    : GaussianNB(),
                    'SVC'   : svm.SVC(kernel='linear', random_state=None),
                    'K-SVM' : svm.SVC(kernel='rbf', random_state=None)
                  }

def isnan(value):
    try:
        import math
        return math.isnan(float(value))
    except:
        return

"""
index：stratifiedで分割したtrain_indexかtest_index
X：要件のリスト
y：ラベルのリスト
textlist：格納する要件のリスト
labellist：格納するラベルのリスト
"""
def loop(index, X, y, textlist, labellist):
    for i in index:
        textlist.append(X[i])
        labellist.append(y[i])

"""
トレーニングに使用された単語をcsvに出力

model：機械学習のモデル
NFR：品質特性
text：要件のリスト
labels：ラベルのリスト
ratio：割合
add_name：品詞指定の時のディレクトリ作るための
count：何回目か
"""
def word_count(model, NFR, text, labels, ratio, add_name, string, count, load_model='NO'):
    df = pd.DataFrame()
    c_0 = Counter()
    c_1 = Counter()
    for line, label in zip(text, labels):
        if label == 0:
            for word in line:
                c_0[word] += 1
        elif label == 1:
            for word in line:
                c_1[word] += 1
    df0 = pd.DataFrame(c_0.most_common(), columns=[NFR, "単語数"])
    df1 = pd.DataFrame(c_1.most_common(), columns=[NFR, "単語数"])
    df = pd.concat([df, df0, df1], axis=1)
    dir_path = '/home/mirai/実験/NFR分類/予測結果/' + model + '_10/' + load_model + '/' + add_name + '/' + NFR + '/' + str(ratio) + '/単語/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    df.to_csv(dir_path + "単語数" + string + "(厚" + str(ratio) + ")_" + str(count) + ".csv")

"""
10-fold cross validation

model：機械学習のモデル
index：どの要件文を使用するかのリスト
docs_list：要件文のリスト
labels_list：ラベルのリスト
NFR：品質特性入れる
ratio：割合
dd_name：品詞指定の時のディレクトリ作るための
Normalization：単語の正規化をするかどうか
"""
def result(model, text_list, label_list, NFR, ratio, add_name, string, Normalization=False):
    global pre_sum, rec_sum, f1_sum
    SKF = StratifiedKFold(n_splits=5, shuffle=True)
    count = 0
    load_model = 'NO'
    if Normalization:
        load_model = 'word2vec'
    for tra_index, te_index in SKF.split(text_list, label_list):
        count += 1
        train_text, test_text = [], []
        train_label, test_label = [], []
        loop(index=tra_index, X=text_list, y=label_list, textlist=train_text, labellist=train_label)
        loop(index=te_index, X=text_list, y=label_list, textlist=test_text, labellist=test_label)



        train = []
        test = []

        for j in range(len(train_text)):  # data_train ▶︎ train
            train.append(train_text[j][1])
        
        word_count(model=model, NFR=NFR, text=train, labels=train_label, ratio=ratio, add_name=add_name, string=string, count=count, load_model=load_model)

        for k in range(len(test_text)):  # data_test ▶︎ test
            test.append(test_text[k][1])

        dense_all_test = []
        dense_all_train = []

        dictionary = corpora.Dictionary(train)

        if Normalization == False:
            docs_train = train
            docs_test = test
            siki = 9999
        # elif Normalization == True:
        #     w1 = wordVecMaker(tokens=train, threshold=siki, nfr=NFR, count=count, classify=classification_model, path=add_path)
        #     docs_train = w1.synonimTransfer(sentences=train, synonyms=w1.get_synonym())
        #     # w2 = get_synonym(test, siki)
        #     docs_test = w1.synonimTransfer(sentences=test, synonyms=w1.get_synonym())

        bow_corpus_train = [dictionary.doc2bow(d) for d in docs_train]
        bow_corpus_test = [dictionary.doc2bow(d) for d in docs_test]

        for bow in bow_corpus_train:
            dense = list(matutils.corpus2dense([bow], num_terms=len(dictionary)).T[0])
            dense_all_train.append(dense)

        for bow2 in bow_corpus_test:
            dense2 = list(matutils.corpus2dense([bow2], num_terms=len(dictionary)).T[0])
            dense_all_test.append(dense2)



        if model == 'SMO':
           prob = svm_problem(train_label, dense_all_train)
           param = svm_parameter("-s 0 -t 0")
           mdl = svmutil.svm_train(prob, param)
           label_predict, accuracy, dec_values = svmutil.svm_predict(test_label, dense_all_test, mdl)
        elif model in modelselection:
            clf = modelselection[model]
            clf.fit(dense_all_train, train_label)
            label_predict = clf.predict(dense_all_test)

        pre_score_ = precision_score(test_label, label_predict, average=None)
        rec_score_ = recall_score(test_label, label_predict, average=None)
        f1_score_ = f1_score(test_label, label_predict, average=None)

        pre_sum += pre_score_[1]
        rec_sum += rec_score_[1]
        f1_sum += f1_score_[1]

        df = pd.DataFrame({'要件': test, '正解': test_label, '予測': label_predict}, columns=['要件', '正解', '予測'])
        dir_path = '/home/mirai/実験/NFR分類/予測結果/' + model + '_10/' + load_model + '/' + add_name + '/' + NFR + '/' + str(ratio) + '/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        df.to_csv(dir_path + '/' + str(siki) + '_' + str(count) + '(厚[' + string + ']).csv')

def main(csv_input, NFR, classification_model, ratio, Normalization=False, verbalnoun=False, phrase=[], load_model='NO'):
    with open('PID_file_10.txt', 'a') as f:
        f.write("\n親{}：子{}".format(os.getppid(), os.getpid()))
    
    plain_text = csv_input["要件"].tolist()
    plain_label = csv_input["最終判断"].tolist()
    labels_ = [label for label in plain_label]
    global pre_sum, rec_sum, f1_sum
    string = "(推移)"
    stop_word = False
    add_name = ''

    if Normalization:
        load_model = 'word2vec'
    
    if len(phrase) == 0:
        add_name = 'なし'
    elif len(phrase) > 0:
        #stop_word = True
        for p in phrase:
            add_name += str(p)
    
    labels = []
    docs = []

    for line in plain_text:
        if verbalnoun:
            docs.append(MecabTokenizer(text=line, rm_sw=stop_word, phrase=phrase).wakati2())
            string = "(推移[する])"
        else:
            docs.append(MecabTokenizer(text=line, rm_sw=stop_word, phrase=phrase).wakati())

    for num, label in enumerate(labels_):
        if isnan(label) != True:
            labels.append((label))

    l_n = []  # 要件の場所
    docs_, labels_ = [], []
    for num, label in enumerate(labels):
        if NFR in label:
            labels_.append((label))
            labels_[num] = 1
            ex = [plain_text[num], docs[num]]
            docs_.append(ex)
            l_n.append(num)

        else:
            labels_.append((label))
            labels_[num] = 0
            ex = [plain_text[num], docs[num]]
            docs_.append(ex)

    """
    SKF = StratifiedKFold(n_splits=10, shuffle=True)
    for train_index, test_index in SKF.split(docs_, labels_):
        d = []
        for i in test_index:
            d.append(labels_[i])
        print(np.bincount(d))
    """

    for i in range(20):
        print(classification_model + '：' + NFR + '：' + str(ratio) + '：' + str(i))
        #text_list, label_list = extract_stratified(text=docs_, label=labels_, ratio=ratio)
        if ratio == 1.0:
            text_list = docs_
            label_list = labels_
        else:
            text_list, test_t, label_list, label_t = train_test_split(docs_, labels_, train_size=ratio, stratify=labels_, random_state=None)
        result(model=classification_model, text_list=text_list, label_list=label_list, NFR=NFR, ratio=ratio, add_name=add_name, string=string, Normalization=Normalization)

    print(classification_model + ':' + NFR + str(ratio))
    pre, rec, f1 = [], [], []
    pre_sum /= 100
    rec_sum /= 100
    f1_sum  /= 100
    pre.append(pre_sum)
    rec.append(rec_sum)
    f1.append(f1_sum)

    df = pd.DataFrame({'Precision': pre, 'Recall': rec, 'F値': f1}, columns=["Precision", "Recall", "F値"])
    dir_path = '/home/mirai/実験/NFR分類/閾値評価/' + classification_model + '_10/' + load_model + '/' + add_name + '/' + NFR + '/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    df.to_csv(dir_path + '[' + add_name + ']' + string + '(厚)_' + str(ratio) + '.csv')

"""
並列処理をするためのメソッド
"""
def wrapper(arg):
    main(*arg)

if __name__ == "__main__":
    with open('PID_file_10.txt', 'w') as f:
        f.write("{}".format(os.getpid()))

    starttime = time.time()
    text = pd.read_csv('dataset.csv')
    text['要件'] = text['要件'].replace(r'[【】]', ' ', regex=True)
    text['要件'] = text['要件'].replace(r'[（）()]', ' ', regex=True)
    text['要件'] = text['要件'].replace(r'[［］\[\]]', ' ', regex=True)
    text['要件'] = text['要件'].replace(r'[@＠]\w+', '', regex=True)
    text['要件'] = text['要件'].replace(r'https?:\/\/.*?[\r\n ]', '', regex=True)
    text['要件'] = text['要件'].replace(r'\n', ' ', regex=True)
    text['要件'] = text['要件'].replace(r'　', '', regex=True)
    text['要件'] = text['要件'].replace(' ', '', regex=True)
    text['要件'] = text['要件'].replace(r'・|/', '、', regex=True)
    text['要件'] = text['要件'].replace(r',', '', regex=True)
    text['要件'] = text['要件'].replace(r'^[0-9]+', '', regex=True)
    text['要件'] = text['要件'].replace(r'[0-9]+', '0', regex=True)
    # stopwords = get_stop_words(text['要件'], n=500, min_freq=100)

    """
    main(csv_input, NFR, classification_model, ratio, Normalization=False, verbalnoun=False, phrase=[], load_model='NO')
    """

    csv_input = text
    NFR = ["機能適合性", "性能効率性", "互換性", "使用性", "信頼性", "セキュリティ", "保守性", "移植性"]
    classification_model = ['NB', 'K-NN', 'SMO']
    num = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    normalization = [False]
    vern = [False]
    phrase = [[], ["名詞"], ["名詞", "動詞"]]
    # phrase = [["名詞"], ["名詞", "動詞"]]
    parameter_list = []

    # for nfr in NFR:
    #     for model in classification_model:
    #         for n in num:
    #             parameter_list.append([csv_input, nfr, model, n, False, False])

    for nfr in NFR:
       for model in classification_model:
           for n in num:
               for ph in phrase:
                   parameter_list.append([csv_input, nfr, model, n, False, False, ph])

    # for nfr in NFR:
    #     for model in classification_model:
    #         for n in num:
    #             for ph in phrase:
    #                 parameter_list.append([csv_input, nfr, model, n, False, True, ph])
    
    with Pool(processes=4) as p:
        p.map(wrapper, parameter_list)

    print(str(time.time() - starttime))
    #main(csv_input = csv_input, NFR='機能適合性', classification_model='SVC', ratio=2, Normalization=False)
