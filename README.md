# NLP
研究で使用していた非機能要件自動分類リポジトリ．
自然言語を扱っているので，自然言語処理がわからない人は，[NLP.md](NLP.md)を参照してください．

JIS25010に基づきラベル付けしたデータセット(dataset.csv)を用いて，機械学習により分類を行う．
preprocessing.pyは形態素解析を行っていて，それ以外の処理をmain.pyで行っています．

## 環境
#### 実行環境
* anaconda3-5.3.1
#### パッケージ
* libsvm
* gensim 3.6.0
* mecab 0.996.2
    * mecab-ipadic-NEologd

## インストール方法
#### LIBSVM
$HOME/buildのようなディレクトリを作り，そこに移動し，以下のコマンドを実行．
```bash
wget http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+tar.gz -O libsvm.tgz
```
解凍
```bash
tar zxvf libsvm.tgz
```
生成されたディレクトリに移動し，以下のコマンドでビルドし終了．
```bash
make
```

#### gensim
```bash
pip install gensim
```

## 使い方
```bash
git clone git@github.com:4AZ77mr/NLP.git
cd NLP
python main.py
```
### 注意点
並列処理をしているので，main.pyの最終行辺りにある **processes=** の値を自分の環境に合わせて設定してください．

## 参考文献
LIBSVM github(https://github.com/cjlin1/libsvm)

LIBSVM使用方法(https://data-science.gr.jp/implementation/iml_libsvm_usage.html)

MeCabインストールMac編(https://qiita.com/paulxll/items/72a2bea9b1d1486ca751)

MeCabインストールWindows編(https://qiita.com/wanko5296/items/eeb7865ee71a7b9f1a3a)

mecab-ipadic-NEologd github(https://github.com/neologd/mecab-ipadic-neologd)

