# 音響解析（環境音・音楽ジャンル判定）システム
収録された音声データから収録された音声が音楽なのか、環境音なのか分析するシステムです。また、音楽であればその音楽がどのジャンルに位置するかを判定します。音響解析システムは**Gradio**を使って使用します。

# ソースコード解説
始めにテスト・学習用の音声データは「GTZAN Dataset」、「ESC-50」、「FREE BGM DOVA-SYNDROME」を使用して$います。
- [peke0157/exp](https://github.com/peke0157/mycode/blob/exp/plot.md)
**app.py**は全ての学習データを反映させ、Gradioを使ってアプリケーションを起動します。
## app.pyの実行方法
app.pyのファイルが存在するディレクトリにアクセスし、ターミナルに下記のコマンドを入力します。
```
$ python app.py
```
上記のコマンドを実行すると、下記のようなコメントが出力されます。
```
* Running on local URL:  http://127.0.0.1:7860
* To create a public link, set `share=True` in `launch()`.
```
URLをCtrl+左クリックでアクセスします。
URLの起動後は写真のようなアプリケーションが表示されます。
<img width="900" height="750" alt="image" src="https://1drv.ms/i/c/f189eb2e8e3a452f/IQBWio6Kx3kvRrps8UpZkaHZAZSbvHn9SGXm0D4G_6tJjU4?e=M7ENtj">
左画面のaudiopathにジャンル判定をする音楽ファイルを入力します。Submitをクリックすると予測結果の音楽ジャンルが出力されます。
# 必要なソフトウェア
- Python

# 必要なライブラリ・モジュール
- numpy（数値計算ライブラリ）
- matplotlib（グラフ描写ライブラリ）
- librosa（音声解析ライブラリ）
- gradio（WebUI作成ライブラリ）
- joblib（並列処理・データ保存ライブラリ）
- pathlib（パス操作を提供するモジュールを含む標準ライブラリ）

# 実験環境
- Ubuntu 24.04.3 LTS
- Python 3.13.5
- librosa 0.11.0
