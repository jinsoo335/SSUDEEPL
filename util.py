import numpy as np
import random

# 분할 비율
train_ratio = 0.6
val_ratio = 0.2

def partition(data_load):
    data = []

    # 훈련 데이터와 정답 데이터를 구분하기 위해
    # dict에서 데이터 가져와서 text, 주제, 점수로 패킹
    for subjcet, content in data_load.data_by_subject.items():
        for text, score in zip(content['text'], content['score']):
            data.append((text, subjcet, score))

    # 데이터 섞기
    random.shuffle(data)

    # 데이터 분할
    train_len = int(len(data) * train_ratio)
    val_len = int(len(data) * val_ratio)

    train_data = data[:train_len]
    val_data = data[train_len: train_len + val_len]
    test_data = data[train_len + val_len:]

    return train_data, val_data, test_data


# 벡터화된 데이터의 길이를 같게 해주기 위한 메소드
def pad_vectors(vectors, max_length):
    padded_vectors = np.zeros((len(vectors), max_length))

    for i, vector in enumerate(vectors):
        length = len(vector)
        padded_vectors[i, :length] = vector

    return padded_vectors