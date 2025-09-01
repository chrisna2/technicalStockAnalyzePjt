"""

>> 랜덤 포레스트(Random Forest)
  - 여러개의 결정트리 모델을 결합하여 만든 모델
  - 다수의 결정 트리를 사용하여 분류나 획귀 작업을 수행하는 앙상블 학습 방법
  - 배깅 기법 (bagging) : 여러 개의 모델을 학습시키고 그 예측을 결합하여 최종 예측을 만드는 방법
  - 특성 무작위성 (feature randomness) : 각 결정 트리를 학습시킬 때 무작위로 선택된 특성의 부분 집합을 사용하여 트리 간의 상관관계를 줄임
  - 위에 두개의 방법을 통해 다수의 결정트리를 생성하고 이들의 예측을 결합하여 최종 예측을 만듦

>> 랜덤 포레스트의 구성요소
  - 부트스트렙 샘플링(Bootstrap Sampling) : 원본 데이터에서 중복을 허용하여 여러 개의 샘플을 무작위로 추출하는 방법
  - 랜덤 피처 선택(Random Feature Selection) : 각 결정 트리를 학습시킬 때 무작위로 선택된 특성의 부분 집합을 사용하여 트리 간의 상관관계를 줄임
  - 결정 트리(Decision Tree) : 각 부트스트렙 샘플과 랜덤 피처를 사용하여 개별 결정 트리를 학습시킴
  - 앙상블 예측(Ensemble Prediction) : 모든 결정 트리의 예측을 결합하여 최종 예측을 만듦
    - 분류 문제의 경우 다수결 투표 방식을 사용
    - 회귀 문제의 경우 평균값을 사용

>> 랜덤 포레스트의 장점
  1) 높은 정확도
    - 여러 개의 결정 트리를 결합하여 예측 성능이 향상됨
    - 일반화 성능이 향상됨
  2) 과적합 방지
    - 무작위 샘플링과 피처 선택을 통해 과적합을 줄임
  3) 다양한 데이터 처리
    - 범주형 및 연속형 데이터를 모두 처리할 수 있음
    - 분류와 회귀 문제에 모두 사용 가능
  4) 특성(feature) 중요도 평가
    - 각 특성이 예측에 얼마나 중요한지 평가할 수 있음
    - 그 평가에 따라 특성 선택(feature selection)에 활용 가능

>> 랜덤 포레스트의 단점
    1) 모델 해석 어려움
     - 여러 개의 결정 트리를 결합하여 복잡한 모델이 되어, 개별 트리의 의사결정 과정을 이해하기 어려움
    2) 계산 비용
     - 많은 수의 결정 트리를 학습시키고 예측하는 데 시간이 많이 걸릴 수 있음
     - 특히 대규모 데이터셋에서는 계산 비용이 높아질 수 있음
    3) 메모리 사용량 많음
     - 모든 결정 트리를 메모리에 저장해야 하므로, 메모리 사용량이 많아질 수 있음
    4) 실시간 예측 어려움, 즉 느림
     - 많은 수의 트리를 통과해야 하므로, 실시간 예측이 필요한 경우 적합하지 않을 수 있음

"""

from sklearn.ensemble import RandomForestRegressor # 랜덤 포레스트 회귀 모델
from sklearn.model_selection import train_test_split # 데이터 분할
import FinanceDataReader as fdr # 금융 데이터 수집 라이브러리
import pandas as pd

if __name__ == '__main__':
    df = fdr.DataReader('005930').dropna()  # 삼성전자, 결측값 제거

    x = [] #
    y = [] #

    for i in range(len(df) - 1):

        a = df.iloc[i].to_numpy() # i번째 행의 모든 열 데이터를 numpy 배열로 변환
        b = df.iloc[i+1]["Close"] # i+1번째 행의 "Close" 열 데이터 (다음날 종가)

        x.append(a)
        y.append(b)

    train_x, test_x, train_y, test_y = train_test_split(x, y)  # 학습 데이터와 테스트 데이터로 분할 (기본 비율 75% 학습, 25% 테스트)

    model = RandomForestRegressor()  # 랜덤 포레스트 회귀 모델 생성m
    model.fit(train_x, train_y)  # 모델 학습
    score = model.score(test_x, test_y)  # 모델 평가 (R^2 점수 계산)

    today_data = df.iloc[-1].to_numpy()  # 가장 최근(오늘)의 주가 데이터
    pred = model.predict([today_data])[0]  # 오늘 데이터를 기반으로 내일 종가 예측
    date = str(df.iloc[-1].name + pd.Timedelta(days=1))[:10]  # 내일 날짜 계산

    print(f"삼성전자 {date} 예측 종가 : {pred:,.0f} (모델 평가 점수 : {score:.4f}) 오차범위는 {(1-score)*100:,.2f}% 입니다.")