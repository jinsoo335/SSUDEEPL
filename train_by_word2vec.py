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
data_load.load_file(['./라벨링_글짓기'])

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


# 모델 초기화
vocab_size = len(word_to_id)
hidden_size = 100
context_size = window_size * 2  # 양쪽 문맥 크기의 합

# 하이퍼 파라미터 설정
learning_rate = 0.001
epochs = 20
batch_size = 50     # 한 번에 처리할 데이터 수


model = SimpleCBOW(vocab_size, hidden_size, context_size)
optimizer = SGD(learning_rate)


for epoch in range(epochs):
    np.random.shuffle(context_target_pairs)
    total_loss = 0
    batch_count = 0

    for i in tqdm(range(0, len(context_target_pairs), batch_size), desc="Contexts Targets process"):
        batch = context_target_pairs[i : i+batch_size]
        contexts, targets = zip(*batch)
        contexts = np.array(contexts)
        targets = np.array(targets)


        loss = model.forward(contexts, targets)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        batch_count += 1

    avg_loss = total_loss / batch_count
    print(f"Epoch {epoch}, Avrage Loss: {avg_loss}")


