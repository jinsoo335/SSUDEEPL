import sys

import util

sys.path.append('../')

from data_load import *
from preprocess import *
from simple_full_connect import *
import numpy as np
from util import *
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

data_load = DataLoad()
data_load.load_file(['./라벨링_글짓기'])

preprocess = Preprocess()

# 데이터 분할
train_data, val_data, test_data = util.partition(data_load)

# 에세이 본문, 주제, 주제의 명료성 점수
X_train, subject_train, y_train = zip(*train_data)
X_val, subject_val, y_val = zip(*val_data)
X_test, subject_test, y_test = zip(*test_data)


# 데이터 전처리
X_total = list(X_train) + list(X_val) + list(X_test)

# 전체 데이터셋에 대해 단어 인덱스 생성
word_to_id, id_to_word = preprocess.create_word_index(X_total)

# 각 데이터셋에 대한 벡터화 수행
train_vectors = preprocess.vectorize(list(X_train), word_to_id)
val_vectors = preprocess.vectorize(list(X_val), word_to_id)
test_vectors = preprocess.vectorize(list(X_test), word_to_id)

# train, val, test 벡터 중 가장 긴 벡터의 길이 찾기
max_length_train = max(len(v) for v in train_vectors)
max_length_val = max(len(v) for v in val_vectors)
max_length_test = max(len(v) for v in test_vectors)

# 가장 긴 벡터의 길이 설정
max_length = max(max_length_train, max_length_val, max_length_test)

# 패딩 설정
train_vectors = pad_vectors(train_vectors, max_length)
val_vectors = pad_vectors(val_vectors, max_length)
test_vectors = pad_vectors(test_vectors, max_length)


# 데이터셋의 통계적 특성 계산 및 출력
def print_dataset_statistics(dataset, name="Dataset"):
    data = np.array(dataset)
    print(f"{name} - 최소값: {np.min(data)}, 최대값: {np.max(data)}, 평균: {np.mean(data)}, 표준편차: {np.std(data)}")

print_dataset_statistics(train_vectors, "Train Vectors")
print_dataset_statistics(val_vectors, "Validation Vectors")
print_dataset_statistics(test_vectors, "Test Vectors")

# 히스토그램 그리기
plt.figure(figsize=(12, 6))
plt.hist(np.array(train_vectors).flatten(), bins=50, alpha=0.5, label='Train')
plt.hist(np.array(val_vectors).flatten(), bins=50, alpha=0.5, label='Validation')
plt.hist(np.array(test_vectors).flatten(), bins=50, alpha=0.5, label='Test')
plt.title("Distribution of Data")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# 하이퍼 파라미터 설정
learning_rate = 0.001
epochs = 20
batch_size = 50     # 한 번에 처리할 데이터 수


# 데이터셋 크기
train_size = len(train_vectors)
val_size = len(val_vectors)
test_size = len(test_vectors)


# 반복 횟수 계산
iter_per_epoch = max(train_size // batch_size, 1)
total_iter = int(epochs * iter_per_epoch)


# 모델 초기화
input_size = max_length
hidden_size = input_size // 10
output_size = 1     # 회귀 문제로 생각해보자 - 출력층 뉴런을 1개로 설정

model = SimpleFullConnect(input_size, hidden_size,output_size)

# # 모델 파라미터 출력
# print("Model Parameters:")
# for layer_name in ['Affine1', 'Affine2', 'Affine3']:
#     W = model.params[f'W{layer_name[-1]}']
#     b = model.params[f'b{layer_name[-1]}']
#     print(f"{layer_name} - W shape: {W.shape}, b shape: {b.shape}")
#
# # 미니배치 데이터 크기 출력
# batch_mask = np.random.choice(train_size, batch_size)
# x_batch = np.array([train_vectors[i] for i in batch_mask])
# y_batch = np.array([y_train[i] for i in batch_mask])
#
# print("\nMini-batch Data Shape:")
# print(f"x_batch shape: {x_batch.shape}")
# print(f"y_batch shape: {y_batch.shape}")
#
# # 호환성 확인
# print("\nCompatibility Check:")
# for layer_name in ['Affine1', 'Affine2', 'Affine3']:
#     W = model.params[f'W{layer_name[-1]}']
#     if x_batch.shape[1] != W.shape[0]:
#         print(f"Incompatibility in {layer_name}: x_batch features {x_batch.shape[1]} != W rows {W.shape[0]}")
#     else:
#         print(f"{layer_name} is compatible.")


# 손실 값 저장
train_loss_list = []
val_loss_list = []

# 학습
for i in tqdm(range(total_iter), desc="Learning"):
    # 미니배치, numpy 배열 형태로 만들어야 함.
    batch_mask = np.random.choice(train_size, batch_size)

    x_batch = np.array([train_vectors[i] for i in batch_mask])
    y_batch = np.array([y_train[i] for i in batch_mask])

    # 순전파 및 손실 계산
    model.loss(x_batch, y_batch)

    # 역전파를 통한 기울기 계산
    grads = model.gradient(x_batch, y_batch)

    # 매개변수 업데이트
    for key in model.params.keys():
        model.params[key] -= learning_rate * grads[key]

    # 학습 과정 출력
    if i % iter_per_epoch == 0:
        train_loss = model.loss(train_vectors, y_train)
        val_loss = model.loss(val_vectors, y_val)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        print(f"Epoch: {i // iter_per_epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

# 테스트 데이터에 대한 성능 평가
test_evaluate = model.calculate_batch_loss(test_vectors, y_test, batch_size)
print(f"Test Accuracy: {test_evaluate}")


# 학습 종료 후 그래프 그리기
epochs = range(len(train_loss_list))
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss_list, 'bo-', label='Training loss')
plt.plot(epochs, val_loss_list, 'r^-', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

