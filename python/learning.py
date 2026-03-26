import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

data_dir = Path('testdata')
npy_files = list(data_dir.glob('**/*.npy'))

X = []
Y = []

for npy_file in npy_files:
    mfccs = np.load(npy_file)
    
    mean = np.mean(mfccs, axis=1)
    std = np.std(mfccs, axis=1)
    concatenate = np.concatenate([mean, std])
    
    genru = npy_file.parent.name
    
    X.append(concatenate)
    Y.append(genru)
    
X = np.array(X)
Y = np.array(Y)

print(X.shape)
print(Y.shape)
print(np.unique(Y))    

# テスト用データと訓練用データに分ける
