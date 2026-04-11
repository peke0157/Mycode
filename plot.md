# FFT前のグラフ
今回は以下のファイル。ディレクトリ構成を例としている
```
mycode
├──python
│  └──test
|     ├──classical
|     |  ├──classical.00000.wav
|     |  ├──classical.00001.wav
|     |  └──classical.00002.wav
│     |
|     ├──environment
│     |  ├──1-977-A-39.wav
│     |  ├──1-1791-A-26.wav
│     |  └──5-9032-A-0.wav
│     |
|     └──hiphop
|        ├──hiphop.00000.wav
|        ├──hiphop.00001.wav
|        └──hiphop.00002.wav
|
└──plot.ipynb etc.
```
必要モジュールをインポートする
```
import numpy as np
import matplotlib.pyplot as plt
import librosa
from pathlib import Path
```
以下のコード
```
base_dir = Path('test')
for audio_dir in base_dir.glob('*.wav'):
    y, sr = librosa.load(audio_dir)

    L = len(y)
    t = np.arange(0,L) / sr
    
    

    plt.plot(t,y)
    plt.title('Audio waveform')
    plt.xlabel('time(s)')
    plt.ylabel('Amplitude')

    plt.show()
```
**Pathlib**のライブラリを使うことでファイル・ディレクトリのパスをオブジェクトとして操作・処理することができる。「base_dir = Path('test')」これはtestディレクトリのパスを設定することでtestディレクトリの中にあるファイルを操作することができる。for文で「test」ディレクトリにある音声ファイルの個数分音声波形を出力している。

# MFCC適用のグラフをプロットするコード
音声・音楽ファイルを機械学習で扱うにはmfccを適用して特徴量を抽出する。**MFCC（メル周波数ケプストラム係数）**とは、音声認識や音声処理において最も広く使用されている特徴量の一つ。MFCCは、音声信号を周波数領域で分析し、人間の耳が音を知覚する方法に近い形で特徴を抽出することを目的としている。これにより、音声認識システムは単なる音の強度やピッチの違いではなく、人間の聴覚の特性を考慮した音声データのパターンを理解することができる。
MFCCのメリットとして、聴覚特性を反映した特徴を抽出することができる点である。単純な周波数分析では、高周波数成分が重要視されすぎることがあるが、MFCCではメルスケールを使用することで、低周波数と高周波数のバランスを適切に調整することができる。
- MFCCを適用したソースコードは下記の通りである。
```
# MFCC適用
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path
import math
import glob

# 保存先のディレクトリ
save_dir = Path('expdata')
base_dir = Path('exp')
for audio_dir in base_dir.glob('**/*.wav'):
    y, sr = librosa.load(audio_dir)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.title('MFCC result')
    plt.xlabel('time(s)')
    plt.ylabel('MFCC Coefficients')
    plt.grid(True)
    plt.colorbar(label='Amplitude(db)')
    plt.tight_layout()
    plt.show()
    
    genru = audio_dir.parent.name
    genru_save_dir = save_dir / genru
    genru_save_dir.mkdir(parents=True, exist_ok=True)
    save_path = genru_save_dir / f"{audio_dir.stem}_mfccs.npy"
    # 保存する
    np.save(save_path, mfccs)
```
- save_dirで抽出した学習済みモデルの保存先ディレクトリを「expdata」にしている。librosaライブラリはmfccするときも便利で、librosaを用いることでmfccの計算が一行で行うことができる。
