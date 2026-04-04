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
**Pathlib**のライブラリを使うことでファイル・ディレクトリのパスをオブジェクトとして操作・処理することができる。「base_dir = Path('test')」これはtestディレクトリのパスを設定することでtestディレクトリの中にあるファイルを操作することができる。