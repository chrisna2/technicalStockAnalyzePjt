"""
>> 주식 가격 예측하기
 -  기본적인 알고리즘 구조
    - 전날의 주식 가격 정보를 통해 다음날의 종가를 맞춤

"""

from sklearn.neighbors import KNeighborsRegressor # KNN 회귀 모델
from sklearn.model_selection import train_test_split # 데이터 분할
import FinanceDataReader as fdr
import pandas as pd
import numpy as np


df = fdr.DataReader('005930').dropna()  # 삼성전자, 결측값 제거

x = []
y = []

for i in range(len(df) - 1):
    a = df.iloc[i].to_numpy()
    b = df.iloc[i+1]["Close"]
    x.append(a)
    y.append(b)

#x = np.array(x)
#y = np.array(y)
#print(x.shape, y.shape)  # (1007, 7) (1007,)

"""
>> K 최근점 이웃 모델 (K-Nearest Neighbors, KNN)
  - 지도학습 중 하나
  - 분류 및 회귀 문제에 모두 사용 가능
  - 단순하고 직관적인 기계학습 알고리즘
  - 새로운 포인트의 클래스를 예측하기 위해 가장 가까운 K개의 이웃을 참조
  - KNN은 데이터 포인트 간의 거리를 계싼하여 새로운 포인트의 클래스에 속하는지 또는 어떤 값을 가질지 예측함
  
>> KNN의 작동 방식
 1. 데이터 포인트의 거리계산
  - 새로운 포인트와 기존 데이터 포인트 간의 거리를 계산
  - 유크리드 거리, 매해튼 거리, 유사도 등을 사용
 2. K개의 최근접 이웃 선택
  - 계산된 거리 값을 기준으로 가장 가까운 K개의 데이터 포인트 선택
 3. 분류 또는 회귀 예측
  - 분류 : K개의 이웃 중 가장 많이 속한 클래스가 새로운 포인트의 클래스
  - 회귀 : K개의 이웃의 평균값이 새로운 포인트의 예측 값

>> KKN의 장점
 - 단순하고 직관적임 : 이해하기 쉽고 구현이 간단하다.
 - 비선형 데이터에 적합 : 복잡한 데이터 분포를 잘 처리할 수 있다.
 - 모델 훈련시간이 따로 필요없음 : 모델 자체는 훈련이 필요없고 예측할떄 연산이 이루어짐
 
>> KNN의 단점
 - 예슥 시간에 지연 : 모든 데이터 포인트 간의 거리를 계산해야되서 데이터양이 많은 경우 예측 시간이 길어질 수 있음
 - 메모리 사용량 많음 : 모든 훈련 데이터를 메모리에 저장해야함
 - 특성 스케일링 필요 : 거리 계산에 민감하므로 각 특성 의 스케일을 맞추는 작업 필요

>> KNN의 K값 선택
 - K가 작을때 : 모델이 데이터의 노이즈에 민감해져 과적합(overfitting)될 수 있음
 - K가 클때 : 모델이 너무 일반화되어 중요한 패턴을 놓칠 수 있음(과소적합, underfitting)
 - 일반적으로 홀수 값을 선택하여 동점 방지
 - 교차검증(cross-validation)을 통해 최적의 K값을 찾는 것이 좋음
 
"""

# 데이터 분할 : 학습세트, 테스트세트로 분할
train_x, test_x, train_y, test_y = train_test_split(x, y)

# KNN 회귀 모델 생성 및 학습
model = KNeighborsRegressor()  # KNN 모델 초기화
model.fit(train_x, train_y) # 학습데이터를 기반으로 모델 학습 
score = model.score(test_x, test_y) # 모델 평가 : 테스트데이터를 사용하여 모델의 성능을 평가, 결정계수를 score에 저장

# 데이터 프레임의 마지막 행(오늘)을 넘파이 배열로 변환하여 today_data에 저장
today_data = df.iloc[-1].to_numpy()
# 모델을 사용하여 내일의 종가를 얘축하고, 그 예측값을 pred에 저장
pred = model.predict([today_data])[0]

# 다음날 날짜 계산 : 마지막 날짜에 하루를 더함
date = str(df.iloc[-1].name + pd.Timedelta(days=1)).split()[0]
# 결과 출력
print(f"{date} 리얼티인컴 예측 종가 : {pred:,.0f} (모델점수 : {score:.4f}) 오차범위는 {(1-score)*100:,.2f}% 입니다.")