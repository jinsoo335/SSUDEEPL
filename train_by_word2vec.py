import sys

import util
from optimizer import SGD

sys.path.append('../')

from data_load import *
from preprocess import *
from simple_full_connect import *
import numpy as np
from util import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from SimpleCBOW import SimpleCBOW

data_load = DataLoad()
data_load.load_file(['./라벨링_샘플'])

preprocess = Preprocess()


data = []
for subjcet, content in data_load.data_by_subject.items():
    for text, score in zip(content['text'], content['score']):
        data.append((text, subjcet, score))

data_texts, data_subjects, data_scores = zip(*data)

word_to_id, id_to_word = preprocess.create_word_index(list(data_texts) + list(data_subjects))


# 벡터화
data_text_vectors = preprocess.vectorize(data_texts, word_to_id)
data_subject_vectors = preprocess.vectorize(data_subjects, word_to_id)


# 문맥과 타겟 단어 데이터 셋 설정
window_size = 2
context_target_pairs = []

for text_vector in data_text_vectors:
    for i in range(window_size, len(text_vector) - window_size):
        context = text_vector[i - window_size:i] + text_vector[i+1 : i+window_size+1]
        target = text_vector[i]
        context_target_pairs.append((context, target))


# 훈련, 검증, 테스트 분할
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# 데이터 인덱스를 무작위로 섞음
indices = np.arange(len(context_target_pairs))
np.random.shuffle(indices)

# 각 데이터셋의 크기 계산
total_size = len(context_target_pairs)
train_size = int(total_size * train_ratio)
val_size = int(total_size * val_ratio)

# 데이터셋 분할
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

# 훈련, 검증, 테스트 데이터셋 생성
train_data = [context_target_pairs[i] for i in train_indices]
val_data = [context_target_pairs[i] for i in val_indices]
test_data = [context_target_pairs[i] for i in test_indices]




# 모델 초기화
vocab_size = len(word_to_id)
hidden_size = 100
context_size = window_size * 2  # 양쪽 문맥 크기의 합

# 하이퍼 파라미터 설정
learning_rate = 0.3
epochs = 20
batch_size = 20    # 한 번에 처리할 데이터 수


model = SimpleCBOW(vocab_size, hidden_size, context_size)
optimizer = SGD(learning_rate)


for epoch in range(epochs):
    np.random.shuffle(train_data)  # 훈련 데이터 섞기
    train_loss = 0
    batch_count = 0

    # 훈련 데이터로 학습
    for i in tqdm(range(0, len(train_data), batch_size), desc="Train process"):
        batch = train_data[i : i + batch_size]
        contexts, targets = zip(*batch)
        contexts = np.array(contexts)
        targets = np.array(targets)

        loss = model.forward(contexts, targets)
        model.backward()
        optimizer.update(model.params, model.grads)
        train_loss += loss
        batch_count += 1

    avg_train_loss = train_loss / batch_count
    print(f"Epoch {epoch}, Training Loss: {avg_train_loss}")

    # 검증 데이터로 성능 평가
    val_loss = 0
    batch_count = 0
    for i in tqdm(range(0, len(val_data), batch_size), desc="val process"):
        batch = val_data[i: i + batch_size]
        contexts, targets = zip(*batch)
        contexts = np.array(contexts)
        targets = np.array(targets)

        loss = model.forward(contexts, targets)
        val_loss += loss
        batch_count += 1

    avg_val_loss = val_loss / batch_count
    print(f"Epoch {epoch}, Validation Loss: {avg_val_loss}")


# 테스트 데이터로 성능 평가
test_loss = 0
batch_count = 0
for i in tqdm(range(0, len(test_data), batch_size), desc="test process"):
    batch = test_data[i: i + batch_size]
    contexts, targets = zip(*batch)
    contexts = np.array(contexts)
    targets = np.array(targets)

    loss = model.forward(contexts, targets)
    test_loss += loss
    batch_count += 1

avg_test_loss = test_loss / batch_count
print(f"Test Loss: {avg_test_loss}")


# for epoch in range(epochs):
#     np.random.shuffle(context_target_pairs)
#     total_loss = 0
#     batch_count = 0
#
#     for i in tqdm(range(0, len(context_target_pairs), batch_size), desc="Contexts Targets process"):
#         batch = context_target_pairs[i : i+batch_size]
#         contexts, targets = zip(*batch)
#         contexts = np.array(contexts)
#         targets = np.array(targets)
#
#         loss = model.forward(contexts, targets)
#         model.backward()
#         optimizer.update(model.params, model.grads)
#         total_loss += loss
#         batch_count += 1
#
#     avg_loss = total_loss / batch_count
#     print(f"Epoch {epoch}, Avrage Loss: {avg_loss}")


