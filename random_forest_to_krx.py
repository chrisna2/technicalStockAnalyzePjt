"""
>> 국내 주식 가운데 랜덤 포레스트 모델을 사용하여 내일 주식 예측

"""
import FinanceDataReader as fdr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import os



if __name__ == '__main__':

    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)

    # KRX 주식 정보 get
    stocks = fdr.StockListing('KRX')

    for s in range(len(stocks)):
        code = stocks.iloc[s]['Code']
        name = stocks.iloc[s]['Name']
        df = fdr.DataReader("NAVER:"+code).dropna()  # 결측값 제거

        x = []
        y = []

        for i in range(len(df)-1):
            a = df.iloc[i].to_numpy()
            b = df.iloc[i+1]['Close']

            x.append(a)
            y.append(b)

        train_x, test_x, train_y, test_y = train_test_split(x,y)

        # 모델 경로 설정
        model_path = os.path.join(model_dir, f'{code}_model.pkl')

        # 모델 로드 또는 학습
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            model = RandomForestRegressor()
            model.fit(train_x, train_y)
            joblib.dump(model, model_path)

        score = model.score(test_x, test_y)
        today_data = df.iloc[-1].to_numpy()
        pred = model.predict([today_data])[0]

        date = str(df.iloc[-1].name + pd.Timedelta(days=1))[:10]

        # 오늘의 종가
        today_price = int(df.iloc[-1]['Close'])

        if today_price < pred :
            print(f"{name}({code})의 현재가 {today_price:,.0f}원 -> {date}일자 주식예측가격은 {pred:,.0f}원 입니다. 오차범위는 {(1-score)*100:,.2f}% 입니다.")

        joblib.dump(model, 'random_forest_model.pkl')
