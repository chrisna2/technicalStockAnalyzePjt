"""

>> XGBoost(Extreme Gradient Boosting) 모델
 - 기계 학습에서 매우 인기있는 그래디언트 부스팅 알고리즘
 - 빠른 학습 속도, 높은 예측 성능, 그리고 다양한 문제 해결 능력이 장점
 - 결정 트리모델의 업그레이드 버전

>> xgboost의 구성 요소
 - 결정트리(base learner)
    - 여러개의 결정트리를 사용하여 학습
    - 각 트리는 약학 학습기(weak learner)로 작동
    - 이들이 모여 강력한 예측 모델을 형성
 - 오차(loss function)
    - 현재 모델의 에측과 실제값 사이 차이를 계산하여 새로운 트리 학습
 - 학습율(learning rate)
    - 각 트리가 모델에 기여하는 정도를 조절
    - 모델이 천천히 학습하도록 함
 - 규제(regularization)
    - 모델의 복잡성을 제어하여 과적합을 방지함
    - L1, L2 규제 등을 포함
 - 부스팅단계(Boosting rounds)
    - 여러 트리를 순차적으로 추가하여 모델 성능을 점진적으로 향상시킴

>> xgboost 작동 방식
 1) 초기 모델 생성
   -  첫번쨰 트리는 전체 데이터의 평균값을 예측하는 간단한 모델로 시작함
 2) 잔여 오차 계산
   -  첫번째 모델의 예측과 실제값 사이에 오차(잔여)를 계산
 3) 잔여에 대한 새로운 트리 학습
   - 오차를 줄이기 위해 새로운 결정 트리를 학습
 4) 모델 결함
   - 기존 모델에 새로 학습된 트리를 추가하여 모델을 업데이트
 5) 반복
   - 원하는 수의 트리가 추가될 때까지 2~4단계를 반복

>> xgboost의 장점
    1) 고성능, 병렬처리
        - 병렬 처리 및 분산 컴퓨팅을 통해 매우 빠른 학습 속도를 자랑함
    2) 높은 얘측 성능
        - 다양한 데이터셋과 문제 유형에서 뛰어난 예측 성능을 보임
    3) 유연성
        - 회귀, 분류, 랭킹 등 다양한 문제에 적용 가능
    4) 과적합 방지
        - 규제 기법을 통해 모델의 복잡성을 제어하여 과적합을 방지함
    5) 확장성
        - 대규모 데이터셋에서도 효율적으로 작동함

>> xgboost의 단점
    1) 복잡성
        - 모델의 하이퍼파라미터가 많아 최적화가 어려울 수 있음
        - 하이퍼파라미터 : 학습률, 트리 깊이, 부스팅 라운드 수 등
    2) 자원 소모
        - 대규모 데이터셋에서는 메모리 및 계산 자원을 많이 소모할 수 있음
    3) 해석 어려움
        - 모델이 복잡하여 개별 트리의 의사결정 과정을 이해하기 어려울 수 있음

"""

from xgboost import XGBRegressor # XGBoost 회귀 모델
from sklearn.model_selection import train_test_split # 데이터 분할
import FinanceDataReader as fdr # 금융 데이터 수집 라이브러리
import pandas as pd

if __name__ == '__main__':
    df = fdr.DataReader('005930').dropna()  # 삼성전자, 결측값 제거

    x = []
    y = []

    for i in range(len(df) - 1):

        a = df.iloc[i].to_numpy() # i번째 행의 모든 열 데이터를 numpy 배열로 변환
        b = df.iloc[i+1]["Close"] # i+1번째 행의 "Close" 열 데이터 (다음날 종가)

        x.append(a)
        y.append(b)

    # 학습 데이터와 테스트 데이터로 분할 (기본 비율 75% 학습, 25% 테스트)
    train_x, test_x, train_y, test_y = train_test_split(x, y)

    model = XGBRegressor() # XGBoost 회귀 모델 객체 생성
    model.fit(train_x, train_y) # 모델 학습

    score = model.score(test_x, test_y) # 모델 평가 (R^2 점수 계산)

    today_data = df.iloc[-1].to_numpy() # 가장 최근(오늘)의 주가 데이터
    pred = model.predict([today_data])[0] # 오늘 데이터를 기반으로 내일 종가 예측

    date = str(df.iloc[-1].name + pd.Timedelta(days=1))[:10]  # 내일 날짜 계산

    # 예측 결과 출력
    print(f"삼성전자 {date} 일자 예상 가격 : {pred:,.0f}원 입니다. 모델 정확도는 {score:.4f}입니다. 오차범위는 {(1-score)*100:,.2f}% 입니다.")