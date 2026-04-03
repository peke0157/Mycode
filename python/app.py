import gradio as gr
import numpy as np
import joblib
import librosa
from pathlib import Path


# 予測関数
def predict(audiopath):
    y, sr = librosa.load(audiopath)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # 2次元配列を変更する処理
    mean = np.mean(mfccs, axis=1)
    std = np.std(mfccs, axis=1)
    features = np.concatenate([mean, std])
    
    # 音楽ジャンル
    class_genru = {
        "blues", "classical", "country", "disco", "environment", "hiphop", "jazz", "metal", "pop", "reggae", "rock"
    }
    # 学習モデルの反映
    scaler = joblib.load('expscaler.pkl')
    model = joblib.load('expmodel.pkl')
    
    transform = scaler.transform([features])
    probs = model.predict_proba([features])[0]
    
    genru = {}
    
    for i, class_genrus in enumerate(class_genru):
        genru[class_genru] = float(probs[i])
        

    return genru


# Grインターフェイスの作成
demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type='filepath'),
    outputs=gr.Label(num_top_classes=11, label='予測ジャンル'),
    title="音楽ジャンル測定アプリ",
)

# アプリケーションの起動
if __name__ == "__main__":
    demo.launch()