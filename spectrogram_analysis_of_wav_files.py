import os
import numpy as np
from scipy.signal import spectrogram
from scipy.io import wavfile
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from PIL import Image

# サンプル音声ファイルのディレクトリを指定
sample_directory = '/content/drive/MyDrive/samples'

# データセットディレクトリ内の音声ファイルを自動的に取得
dataset_directory = '/content/drive/MyDrive/dataset'

# スペクトログラム画像の保存ディレクトリを指定（サンプル音声とデータセット用）
sample_spectrogram_directory = '/content/drive/MyDrive/samples/spectrograms'
dataset_spectrogram_directory = '/content/drive/MyDrive/dataset/spectrograms'

# スペクトログラム画像保存ディレクトリが存在しない場合は作成
os.makedirs(sample_spectrogram_directory, exist_ok=True)
os.makedirs(dataset_spectrogram_directory, exist_ok=True)

# サンプル音声ファイルの一覧を取得
sample_files = [f for f in os.listdir(sample_directory) if f.endswith('.wav')]

# データセットディレクトリ内の音声ファイルの一覧を取得
dataset_files = [f for f in os.listdir(dataset_directory) if f.endswith('.wav')]

# サンプル音声ファイルごとに処理
for sample_file in sample_files:
    sample_word = os.path.splitext(sample_file)[0]

    # スペクトログラム画像のファイルパスを構築（サンプル音声用）
    sample_spectrogram_filename = os.path.join(sample_spectrogram_directory, f'{sample_word}_spectrogram.png')

    # すでにスペクトログラム画像が存在する場合はスキップ
    if os.path.exists(sample_spectrogram_filename):
        continue

    # サンプル音声ファイルの読み込み
    sample_sr, sample_audio = wavfile.read(os.path.join(sample_directory, sample_file))

    # スペクトログラムを計算
    f_sample, t_sample, S_sample = spectrogram(sample_audio, fs=sample_sr)

    # スペクトログラム画像を保存
    plt.figure(figsize=(10, 5))
    plt.imshow(10 * np.log10(S_sample), cmap='viridis', origin='lower', aspect='auto')
    plt.axis('off')
    plt.savefig(sample_spectrogram_filename, bbox_inches='tight', pad_inches=0)
    plt.close()

# データセット音声ファイルごとに処理
for dataset_file in dataset_files:
    dataset_word = os.path.splitext(dataset_file)[0]

    # スペクトログラム画像のファイルパスを構築（データセット用）
    dataset_spectrogram_filename = os.path.join(dataset_spectrogram_directory, f'{dataset_word}_spectrogram.png')

    # すでにスペクトログラム画像が存在する場合はスキップ
    if os.path.exists(dataset_spectrogram_filename):
        continue

    # データセット音声ファイルの読み込み
    dataset_sr, dataset_audio = wavfile.read(os.path.join(dataset_directory, dataset_file))

    # スペクトログラムを計算
    f_dataset, t_dataset, S_dataset = spectrogram(dataset_audio, fs=dataset_sr)

    # スペクトログラム画像を保存
    plt.figure(figsize=(10, 5))
    plt.imshow(10 * np.log10(S_dataset), cmap='viridis', origin='lower', aspect='auto')
    plt.axis('off')
    plt.savefig(dataset_spectrogram_filename, bbox_inches='tight', pad_inches=0)
    plt.close()

# サンプル音声ファイルごとに処理（画像からコサイン類似度を計算）
for sample_file in sample_files:
    sample_word = os.path.splitext(sample_file)[0]

    # サンプルスペクトログラム画像のファイルパスを構築
    sample_spectrogram_filename = os.path.join(sample_spectrogram_directory, f'{sample_word}_spectrogram.png')

    # サンプルスペクトログラム画像を読み込み
    sample_spectrogram = Image.open(sample_spectrogram_filename)

    # データセット音声ファイルごとに処理
    for dataset_file in dataset_files:
        dataset_word = os.path.splitext(dataset_file)[0]

        # データセットスペクトログラム画像のファイルパスを構築
        dataset_spectrogram_filename = os.path.join(dataset_spectrogram_directory, f'{dataset_word}_spectrogram.png')

        # データセットスペクトログラム画像を読み込み
        dataset_spectrogram = Image.open(dataset_spectrogram_filename)

        # コサイン類似度を計算して類似性を評価
        similarity = cosine_similarity(np.array(sample_spectrogram).reshape(1, -1), np.array(dataset_spectrogram).reshape(1, -1))

        # 類似性に基づいて反応
        similarity_threshold = 0.99  # 閾値の設定

        if similarity[0, 0] >= similarity_threshold:
            print(f'サンプル "{sample_word}" に対する反応が検出された単語: {dataset_word}')
