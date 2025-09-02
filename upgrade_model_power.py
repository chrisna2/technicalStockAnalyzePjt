"""
>> 모델 성능 높이기
"""
from xgboost import XGBClassifier # XGBoost 분류 모델
from sklearn.model_selection import train_test_split # 데이터 분할
from sklearn.metrics import classification_report # 분류 성능 평가
from tqdm import tqdm # 진행률 표시줄
import FinanceDataReader as fdr # 금융 데이터 수집 라이브러리
import pandas as pd
import os
import joblib

def calculate_rsi(data, window):
    # 갸격 변화
    delta = data.diff()

    # 가격의 상승과 하강을 분리
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # 평균 이득과 평균 손실 계산
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # 상대 강도 계산
    rs = avg_gain / avg_loss  # 0으로 나누는 것을 방지하기 위해 작은 값 추가

    # RSI 계산
    rsi = 100 - (100 / (1 + rs))

    # RSI 값이 NaN인 경우 0으로 채우기
    return rsi

#메인
if __name__ == '__main__':

    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)

    # 모델 경로 설정
    model_path = os.path.join(model_dir, 'upgrade_model_power.pkl')
    model = None # 모델 초기화

    stocks = fdr.StockListing('KRX')  # KRX 종목 전체

    # 모델 로드 또는 학습
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        window = 100  # 이동평균 기간

        x = []  #
        y = []  #

        # 모든 종목에 대해 반복
        for s in tqdm(range(len(stocks))):
            code = stocks.iloc[s]['Code']
            name = stocks.iloc[s]['Name']
            # 주가 데이터 수집 , 네이버, 결측값 제거
            df = fdr.DataReader("NAVER:" + code).dropna()
            df["RSI"] = calculate_rsi(df["Close"], 14)  # RSI 추가 , 2주
            df = df.dropna()  # RSI 계산 후 생긴 결측값 제거

            for i in range(len(df) - window):
                # i번째 행부터 window 길이만큼의 "RSI"와 "Change" 열 데이터를 numpy 배열로 변환 후 1차원 배열로 평탄화
                a = df.iloc[i: i + window][["RSI", "Change"]].to_numpy().flatten()
                # i+window번째 행의 "Close" 열 데이터 (다음날 종가)
                b = int(df.iloc[i + window - 1]["Close"] * 1.05 <= df.iloc[i + window]["Close"])

                x.append(a)
                y.append(b)

        # 학습 데이터와 테스트 데이터로 분할 (기본 비율 75% 학습, 25% 테스트)
        train_x, test_x, train_y, test_y = train_test_split(x, y)
        # XGBoost 회귀 모델 객체 생성
        model = XGBClassifier()
        model.fit(train_x, train_y)  # 모델 학습
        report = classification_report(test_y, model.predict(test_x)) # 분류 성능 평가
        print(report)
        joblib.dump(model, model_path)  # 모델 저장

    for s in range(len(stocks)):
        code = stocks.iloc[s]['Code']
        name = stocks.iloc[s]['Name']
        # 주가 데이터 수집 , 네이버, 결측값 제거
        df = fdr.DataReader("NAVER:"+code).dropna()
        # RSI 추가
        df["RSI"] = calculate_rsi(df["Close"], 14) # RSI 추가 , 2주
        df = df.dropna()  # RSI 계산 후 생긴 결측값 제거

        today_data = df.iloc[-window:][["RSI", "Change"]].to_numpy().flatten()  # 가장 최근(오늘)의 주가 데이터

        try:
            pred = model.predict([today_data])[0]  # 오늘 데이터를 기반으로 내일 종가 예측
        except:
            continue

        if pred:
            print(f"5% 급등 종목 발견!! : {name}({code})")


"""

              precision    recall  f1-score   support

           0       0.78      0.94      0.85       265
           1       0.52      0.20      0.28        87

    accuracy                           0.76       352
   macro avg       0.65      0.57      0.57       352
weighted avg       0.71      0.76      0.71       352

>> precision : 모델이 양성(주가 상승)으로 예측한 것 중 실제로 양성인 비율
>> recall : 실제 양성 중 모델이 양성으로 정확히 예측한 비율
>> accuracy : 전체 예측 중 맞춘 비율
>> f1-score : precision과 recall의 조화 평균
>> support : 각 클래스의 실제 샘플 수
>> macro avg : 클래스별 성능 지표의 단순 평균
>> weighted avg : 클래스별 성능 지표의 샘플 수로 가중 평균

>> 0 : 주가가 5% 이상 상승하지 않음 -> 78% 확률로 맞춤
>> 1 : 주가가 5% 이상 상승함-> 52% 확률로 맞춤

>>> 모델 성능이 향상된 이유
    1) 특징 엔지니어링
        - 단순히 원시 주가 데이터만 사용하는 대신, RSI와 같은 기술적 지표를 추가하여 모델이 더 많은 정보를 학습할 수 있도록 함
    2) 시계열 데이터 활용
        - 이동평균 기간(window)을 사용하여 과거의 여러 시점의 데이터를 하나의 입력으로 사용함으로써, 시간에 따른 패턴을 더 잘 포착할 수 있게 됨
    3) 모델 선택
        - XGBoost는 강력한 성능과 과적합 방지 기능을 갖춘 모델로, 복잡한 데이터에서도 좋은 예측 성능을 보임
    4) 데이터 양 증가
        - 여러 종목의 데이터를 함께 사용함으로써 학습 데이터의 양이 증가하여 모델이 더 일반화된 패턴을 학습할 수 있게 됨
    5) 하이퍼파라미터 튜닝 가능성
        - XGBoost의 다양한 하이퍼파라미터를 조정하여 모델의 성능을 최적화할 수 있는 가능성이 있음




"""