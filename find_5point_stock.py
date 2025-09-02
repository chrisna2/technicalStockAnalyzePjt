"""
> 5% 급등 종목 찾기

- 목표 : 내일 종가가 오늘 종가보다 5% 이상 상승할 종목 찾기

>> 기존의 회귀 모델 vs 분류 모델 비교
- 회귀 모델 : 내일 종가를 직접 예측, 예측값이 실제값과 얼마나 가까운지 평가 (R^2 점수)
- 분류 모델 : 내일 종가가 오늘 종가보다 5% 이상 상승하는지 여부를 예측, 예측값이 실제값과 얼마나 일치하는지 평가 (정확도, 정밀도, 재현율 등)
- 분류 모델이 더 직관적이고 실용적일 수 있음, 특히 투자 결정에 있어 상승 여부가 중요할 때
- 분류 모델은 불균형한 클래스 문제에 민감할 수 있음, 예를 들어 상승하는 종목이 매우 적은 경우



"""
from sklearn.ensemble import RandomForestClassifier # 랜덤 포레스트 분류 모델
from sklearn.model_selection import train_test_split # 데이터 분할
from sklearn.metrics import classification_report # 분류 성능 평가
import FinanceDataReader as fdr # 금융 데이터 수집 라이브러리
import pandas as pd

if __name__ == '__main__':

    df = fdr.DataReader('005930').dropna()  # 삼성전자, 결측값 제거

    x = [] #
    y = [] #

    for i in range(len(df) - 1):

        a = df.iloc[i].to_numpy() # i번째 행의 모든 열 데이터를 numpy 배열로 변환
        b = int(df.iloc[i]["Close"] * 1.05 <= df.iloc[i+1]["Close"]) # i+1번째 행의 "Close" 열 데이터 (다음날 종가)

        x.append(a)
        y.append(b)

    train_x, test_x, train_y, test_y = train_test_split(x, y)

    model = RandomForestClassifier() # 랜덤 포레스트 분류 모델 객체 생성
    model.fit(train_x, train_y) # 모델 학습

    report = classification_report(test_y, model.predict(test_x)) # 분류 성능 평가

    print(report)

    today_data = df.iloc[-1].to_numpy() # 가장 최근(오늘)의 주가 데이터
    date = str(df.iloc[-1].name + pd.Timedelta(days=1))[:10]  # 내일 날짜 계산

    pred = model.predict([today_data])[0] # 오늘 데이터를 기반으로 내일 종가 예측

    if pred:
        print(f"삼성전자 {date} 일자 주가가 5% 이상 상승할 것으로 예측됩니다.")
    else:
        print(f"삼성전자 {date} 일자 주가가 5% 이상 상승하지 않을 것으로 예측됩니다.")

    
"""

              precision    recall  f1-score   support

           0       0.99      1.00      1.00       746
           1       0.00      0.00      0.00         4

    accuracy                           0.99       750
   macro avg       0.50      0.50      0.50       750
weighted avg       0.99      0.99      0.99       750

>> accuracy(정확도) : 전체 예측 중에서 맞게 예측한 비율, 모델의 전체 예측 가운데 맞춘 비율
>> Precision(정밀도) : 양성으로 예측한 것 중에서 실제로 양성인 비율, 모델이 양성으로 예측한 것 중에서 실제로 양성인 비율
    - 모델이 1로 예측한 것 가운데 실제 1인 것의 비율
>> Recall(재현율) : 실제 양성 중에서 모델이 양성으로 예측한 비율, 실제 양성인 것 중에서 모델이 양성으로 예측한 비율
    - 실제 1인 것 가운데 모델이 1로 예측한 것의 비율
>> 클래스 0 : (5% 이상 상승하지 않음) -> 그렇게 예측 한 경우 맞출 확률 -> 99%
>> 클래스 1 : (5% 이상 상승함) -> 그렇게 예측 한 경우 맞출 확률 -> 0%
"""