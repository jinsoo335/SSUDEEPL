#!/usr/bin/env python
# coding: utf-8

# ### 에세이 데이터 로드 클래스
# 
# - 에세이 라벨링 데이터에서 에세이 본문, 주제, 주제의 명료성 점수를 추출
# - 주제의 명료성 점수 같은 경우 3명의 평가자가 각각 점수를 매기기에 평균 점수 값을 가져온다.

# In[ ]:
import os
import json
from preprocess import *
from tqdm import tqdm


class DataLoad:
    def __init__(self):
        self.data_by_subject = {}

    def load_file(self, folder_path_list):

        preprocess = Preprocess()

        for folder_path in tqdm(folder_path_list, desc="DataLoad", leave=False):

            for file in tqdm(os.listdir(folder_path), desc="ReadFile", leave=False):

                if file.endswith('.json'):
                    with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        # 텍스트 추출
                        text = data['paragraph'][0]['paragraph_txt']

                        text = preprocess.preprocessing(text)

                        # 주제 추출
                        subject = data['info']['essay_main_subject']

                        subject = preprocess.preprocessing(subject)

                        # 주제의 명료성 점수 추출
                        score = 0
                        for i in range(3):
                            score += data['score']['essay_scoreT_detail']['essay_scoreT_cont'][i][0]

                        score /= 3

                        if subject not in self.data_by_subject:
                            self.data_by_subject[subject] = {
                                'text': [],
                                'score': []
                            }

                        self.data_by_subject[subject]['text'].append(text)
                        self.data_by_subject[subject]['score'].append(score)
