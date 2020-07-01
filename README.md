# NLP
研究で使用していた非機能要件自動分類リポジトリ．非機能要件に関しては[non-functional.md](non-functional.md)を参照してください．
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
以下の通りに実行していく．
まず，以下のコマンドでディレクトリの作成を行う．ディレクトリ名(buildの部分)は任意．
```bash
mkdir build
```
次に，以下のコマンドで作成してディレクトリに移動する．
```bash
cd build
```
そして，以下のコマンドを実行することでファイルをダウンロードする．
```bash
wget http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+tar.gz -O libsvm.tgz
```
以下のコマンドで解凍．
```bash
tar zxvf libsvm.tgz
```
上記のコマンドを実行した際に生成されたディレクトリに移動し，以下のコマンドでビルドし終了．
```bash
make
```

#### gensim
ターミナルで以下のコマンドを実行してgensimをインストール．
```bash
pip install gensim
```

## 使い方
ターミナルで以下を実行．
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

ビルドとデプロイ(https://qiita.com/isoyam/items/3d1fc5cf7403cdf4818d)

