import re
import sys
import unicodedata
import nltk
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

file = '-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd'

#クリーニング
def clean_text(text):
    replaced_text = text.replace(r'[【】]', ' ', regex=True)
    replaced_text = replaced_text.replace(r'[（）()]', ' ', regex=True)
    replaced_text = replaced_text.replace(r'[［］\[\]]', ' ', regex=True)
    replaced_text = replaced_text.replace(r'[@＠]\w+', '', regex=True)
    replaced_text = replaced_text.replace(r'https?:\/\/.*?[\r\n ]', '', regex=True)
    replaced_text = replaced_text.replace(r'\n', ' ', regex=True)
    replaced_text = replaced_text.replace(r'　', '', regex=True)
    replaced_text = replaced_text.replace(' ', '', regex=True)
    replaced_text = replaced_text.replace(r'・|/', '、', regex=True)
    replaced_text = replaced_text.replace(r',', '', regex=True)
    replaced_text = replaced_text.replace(r'^[0-9]+', '', regex=True)
    replaced_text = replaced_text.replace(r'[0-9]+', '0', regex=True)
    return replaced_text

#形態素解析(mecab)
class MecabTokenizer(object):
    def __init__(self, text, rm_sw=True, phrase=[]):
        self.text = text
        self.rm_sw = rm_sw      #trueならストップワード除去
        self.phrase = phrase

    def wakati(self):
        import MeCab

        mecab = MeCab.Tagger(file)
        mecab.parse('')
        node = mecab.parseToNode(self.text)
        word_list = []
        wakati_list = []
        s_w_pos = ['BOS/EOS', "記号"]

        while node:
            word = node.surface                 #表層形を取得
            pos = node.feature.split(',')[0]    #品詞を取得
            if pos in s_w_pos:                  #記号，文書の始まりと終わりの場合""
                word = ""
            elif len(self.phrase) == 0:           #品詞を設定しない場合
                word_list.append(word)
            else:
                if pos in self.phrase:          #品詞を設定した場合，その品詞のみをリストに追加
                    word_list.append(word)
            node = node.next

        if self.rm_sw:
            wakati_list.append(' '.join(word_list)) #rem_stopwordsに突っ込むための処理
            return rem_stopwords(wakati_list)       #ストップワードを消してる
        else:
            return word_list

    """
    語幹が「する」のときに結合
    """
    def wakati2(self):
        import MeCab

        mecab = MeCab.Tagger(file)
        mecab.parse('')
        node = mecab.parseToNode(self.text)
        word_list = []
        wakati_list = []
        s_w_pos = ['BOS/EOS', "記号"]
        pre_word = ""

        while node:
            word = node.surface
            pos = node.feature.split(',')[0]
            if pos in s_w_pos:  # 記号，文書の始まりと終わりの場合""
                word = ""
            elif len(self.phrase) == 0:  # 品詞を設定しない場合
                if node.feature.split(',')[6] == "する":
                    word_list.remove(pre_word)
                    word = pre_word + word
                pre_word = word
                word_list.append(word)
            else:
                if pos in self.phrase:  #品詞を設定した場合，その品詞のみをリストに追加
                    if node.feature.split(',')[6] == "する":
                        word_list.remove(pre_word)
                        word = pre_word + word
                    pre_word = word
                    word_list.append(word)
            node = node.next

        if self.rm_sw:
            wakati_list.append(' '.join(word_list))  # rem_stopwordsに突っ込むための処理
            return rem_stopwords(wakati_list)  # ストップワードを消してる
        else:
            return word_list


#形態素解析(nagisa)
class NagisaTokenizer(object):
    def __init__(self, phrase=[]):
        self.phrase = phrase

    def tokenize(self, text):
        import nagisa
        
        wakati = []
        #         tagging_words = nagisa.tagging(text)
        #         words = tagging_words.words
        # ngs = nagisa.Tagger()#single_word_list=['ハローワーク'])
        words = normalize(text)
        words = nagisa.wakati(text)
        postags = nagisa.decode(words)
        pos_index = [i for i, x in enumerate(postags) if x in self.phrase]
        ex_words = [words[i] for i in pos_index]
        word_df = pd.DataFrame(ex_words, columns=['word'])
        word_df = word_df['word'].replace(r'^[あ-ん]+$', '', regex=True)
        word_df = word_df[word_df != '']
        words = word_df.tolist()
        words = ' '.join(map(str, words))

        # postags = limit_words.postags
        wakati.extend([words])
        return rem_stopwords(wakati)  # ex_words


#単語の正規化
def normalize(text):
    normalized_text = normalize_unicode(text)
    normalized_text = lower_text(normalized_text)
    return normalized_text

def lower_text(text):
    return text.lower()

def normalize_unicode(text, form='NFKC'):
    normalized_text = unicodedata.normalize(form, text)
    return normalized_text

def lemmatize_term(term, pos=None):
    if pos is None:
        synsets = wordnet.synsets(term)
        if not synsets:
            return term
        pos = synsets[0].pos()
        if pos == wordnet.ADJ_SAT:
            pos = wordnet.ADJ
    return nltk.WordNetLemmatizer().lemmatize(term, pos=pos)

"""
この二つはネットから拾った．(https://qiita.com/chamao/items/7edaba62b120a660657eの辞書による方式)
"""
#ストップワード取得
def create_stopwords():
    stop_words = []
    with open('/Users/mirai/実験/PycharmProjects/classifier/stopwords/stop_words.txt', mode='r') as f:
        for word in f:
            word = word.replace('\n', '')
            if len(word) > 0:
                stop_words.append(word)
    return stop_words

#ストップワード除去
def rem_stopwords(text):
    global i
    stop_words = create_stopwords()
    vectorizer = CountVectorizer(stop_words=stop_words, token_pattern=u'(?u)\\b\\w+\\b') #token_patternで一文字のみも含むように設定
    #vectorizer = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b', stop_words=stop_words)
    #vectorizer = TfidfVectorizer(stop_words=stop_words)

    tfidf = vectorizer.fit_transform(text)

    return vectorizer.get_feature_names()

if __name__ == "__main__":
    text = "thinclient端末0，0台0台"
    te = re.search(r'[0-9]\，[0-9]', text)
    print(te)
    # print(MecabTokenizer(text=text, rm_sw=False).wakati())
    # print(MecabTokenizer(text=text).wakati())
    # print(MecabTokenizer(text=text, rm_sw=False, phrase=["名詞"]).wakati())
    # print(MecabTokenizer(text=text, phrase=["名詞"]).wakati())
    # print(MecabTokenizer(text=text, rm_sw=False, phrase=["名詞", "動詞"]).wakati())
    # print(MecabTokenizer(text=text, phrase=["名詞", "動詞"]).wakati())
    # print("する")
    # print(MecabTokenizer(text=text, rm_sw=False).wakati2())
    # print(MecabTokenizer(text=text).wakati2())
    # print(MecabTokenizer(text=text, rm_sw=False, phrase=["名詞"]).wakati2())
    # print(MecabTokenizer(text=text, phrase=["名詞"]).wakati2())
    # print(MecabTokenizer(text=text, rm_sw=False, phrase=["名詞", "動詞"]).wakati2())
    # print(MecabTokenizer(text=text, phrase=["名詞", "動詞"]).wakati2())
    """
    text = pd.read_csv('./../厚生労働省.csv')
    mecab = []
    for line in text["要件"].tolist():
        mecab.append(MecabTokenizer(text=line).wakati())
    #print(mecab)
    """
    """
    nagisa = []
    for line in text["要件"].tolist():
        nagisa.append(NagisaTokenizer().tokenize(text=line))
    print(nagisa)
    """