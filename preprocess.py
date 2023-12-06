#!/usr/bin/env python
# coding: utf-8

# ### 전처리 수행 클래스
# 
# - 문장 구분 제거
# - 특수 문자 제거(공백으로 분리)
# - 어간 추출(Okt 사용)
# - 불용어 제외(다만, 이 경우 제외하고 제외하지 않고를 둘 다 확인해야할 필요가 있다.)

# In[3]:


from konlpy.tag import Okt
import re


class Preprocess:
    def __init__(self):
        self.analyzer = Okt()
        self.stopwords = self.load_stopwords('stop_words.txt')


    def tf_idf_vectorizer(self, texts):
        print()


    # 단어를 인덱스로
    def create_word_index(self, texts):
        word_to_id = {}
        id_to_word = {}

        for text in texts:
            for word in text.split():
                if word not in word_to_id:
                    new_id = len(word_to_id)
                    word_to_id[word] = new_id
                    id_to_word[new_id] = word

        return word_to_id, id_to_word


    # 벡터화를 통해 숫자 데이터로 변경
    def vectorize(self, texts, word_to_id):
        vector_texts = []

        for text in texts:
            vector = []

            for word in text.split():
                if word in word_to_id:
                    vector.append(word_to_id[word])
            vector_texts.append(vector)

        return vector_texts

    def preprocessing(self, text):
        # 문장 구분 제거
        text = re.sub(r'#@문장구분#', '', text)

        # 특수 문자 제거(공백으로 분리)
        text = re.sub(r'[^가-힣a-zA-Z0-9 ]', ' ', text)

        # 어간 추출 (생각함, 생각합니다. -> 생각한다)
        #words = self.analyzer.morphs(text, stem=True)

        # 형태소 분석
        words = self.analyzer.morphs(text)

        # 불용어 제외
        # words = [word for word in words if word not in self.stopwords]

        return ' '.join(words)





    # 로딩해온 문장들을 전처리 과정을 거친 후, 벡터화 시킨다.
    def preprocessing_list(self, texts):
        for i in range(len(texts)):
            texts[i] = self.preprocessing(texts[i])

        word_to_id, id_to_word = self.create_word_index(texts)
        vectors = self.vectorize(texts, word_to_id)
        return vectors, word_to_id, id_to_word



    def load_stopwords(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            stopwords = [line.strip() for line in file.readlines()]
        return stopwords
