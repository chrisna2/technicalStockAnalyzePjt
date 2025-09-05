import FinanceDataReader as fdr  # 주식 데이터를 가져오기 위한 라이브러리
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split  # 데이터를 학습용/테스트용으로 나누는 함수
import pandas as pd  # 데이터 분석 및 처리를 위한 라이브러리
import joblib  # 학습된 모델 저장/불러오기 위한 라이브러리
import os  # 파일 및 디렉토리 작업을 위한 라이브러리
from datetime import datetime  # 날짜 및 시간 처리를 위한 라이브러리
from openpyxl import load_workbook  # 엑셀 파일 작업을 위한
from tqdm import tqdm  # 진행 상황 표시를 위한 라이브러리
import numpy as np  # 수치 계산을 위한 라이브러리

if __name__ == '__main__':
    model_dir = 'models'  # 모델 저장 디렉토리
    data_dir = 'datas'  # 모델 데이터 저장 디렉토리
    excel_dir = 'results'  # 결과 엑셀 파일 저장 디렉토리

    os.makedirs(model_dir, exist_ok=True)  # 디렉토리가 없으면 생성
    os.makedirs(data_dir, exist_ok=True)  # 데이터 디렉토리가 없으면 생성
    os.makedirs(excel_dir, exist_ok=True)

    # KRX 종목 데이터 가져오기
    stocks = fdr.StockListing('KRX').head(10) # 상위 100개 종목만 사용, 사유 속도 이슈

    """
    # 내가 가지고 있는 종목 데이터로 테스트
    stocks_list = [
        {"code": "005930", "market": "KRX"},  # 삼성전자 (한국)
        {"code": "AAPL", "market": "NASDAQ"},  # 애플 (미국)
        {"code": "000660", "market": "KRX"},  # SK하이닉스 (한국)
        {"code": "MSFT", "market": "NASDAQ"}  # 마이크로소프트 (미국)
    ]
    """

    # 결과 저장 리스트
    results = []

    # 각 종목에 대해 데이터 처리 및 예측 수행 (진행 상황 표시)
    for s in tqdm(range(len(stocks)), desc="Processing stocks"):
        code = stocks.iloc[s]['Code']  # 종목 코드
        name = stocks.iloc[s]['Name']  # 종목 이름

        # 모델 로드 또는 학습
        model_path = os.path.join(model_dir, f'{code}_model.pkl') # 모델 명의 저장은 종목의 코드를 기준으로 생성
        data_path = os.path.join(data_dir, f'{code}_data.npz')

        """
        random_forest_to_krx_study.py에서 과거 주가 데이터는 
        fdr.DataReader("NAVER:" + code)를 통해 가져옵니다. 
        이 함수는 기본적으로 해당 종목의 최초 상장일부터 현재 날짜까지의 데이터를 반환합니다.  
        따라서, 데이터의 시작일은 해당 종목의 상장일이고, 종료일은 스크립트를 실행한 날짜입니다. 
        
        """

        if os.path.exists(model_path) and os.path.exists(data_path):
            # Fetch today's data
            today = datetime.now().strftime('%Y-%m-%d')
            # print(f"today : {today}")
            df = fdr.DataReader("NAVER:" + code, start=today, end=today).dropna()  # 결측값 제거
        else:
            df = fdr.DataReader("NAVER:" + code).dropna()  # 결측값 제거, 전체 일자 데이터 가져 오기
        
        x = []
        y = []

        for i in range(len(df)-1):
            a = df.iloc[i].to_numpy()
            b = df.iloc[i+1]['Close']
            x.append(a)
            y.append(b)

        # model = joblib.load(model_path) if os.path.exists(model_path) else None  # 모델 로드
        if os.path.exists(model_path) and os.path.exists(data_path):
            model = joblib.load(model_path)
            data = np.load(data_path)
            old_x, old_y = data['x'], data['y']
        else:
            model = RandomForestRegressor()
            old_x, old_y = np.empty((0, 6)), np.empty(0)  # Initialize empty arrays for new training

        # X가 비어있을 경우에도 shape 맞춰주기
        if len(x) == 0:
            x = np.empty((0, old_x.shape[1]))

        # Y가 비어있을 경우에도 shape 맞춰주기
        if len(y) == 0:
            y = np.empty((0,))

        # 기존 데이어와 새로운 데이터 결합철
        combined_x = np.vstack((old_x, x))
        combined_y = np.hstack((old_y, y))

        # 모델 재학습
        model.fit(combined_x, combined_y)

        # 데이터를 학습용과 테스트용으로 분리
        train_x, test_x, train_y, test_y = train_test_split(combined_x, combined_y)

        # 모델평가
        score = model.score(test_x, test_y)

        # 모델 및 데이터 저장
        joblib.dump(model, model_path)
        np.savez(data_path, x=combined_x, y=combined_y)

        # 예측 수행
        today_data = df.iloc[-1].to_numpy()
        pred = model.predict([today_data])[0]

        # 내일 일자 계산
        date = str(df.iloc[-1].name + pd.Timedelta(days=1))[:10]  # 내일 날짜

        # 오늘의 종가
        today_price = int(df.iloc[-1]['Close'])

        # 예측 결과가 오늘 종가보다 높을 경우에만 결과 저장
        if today_price < pred:
            results.append({
                "종목명": name,
                "종목코드": code,
                "현재종가": today_price,
                "내일_예측가": pred,
                "상승율": str((pred - today_price) / today_price * 100) + '%',
                "오차범위(%)": (1 - score) * 100,
            })

    # 결과를 엑셀 파일로 저장
    if results:
        today = datetime.now()  # 현재 날짜
        month_year = today.strftime('%Y년 %m월')  # 파일명에 사용할 연도와 월
        file_name = os.path.join(excel_dir, f'{month_year} KRX예측.xlsx')  # 엑셀 파일 경로
        sheet_name = today.strftime('%Y-%m-%d')  # 시트명에 사용할 날짜

        df = pd.DataFrame(results)  # 결과를 데이터프레임으로 변환

        # 엑셀 파일에 저장 (파일이 있으면 추가, 없으면 새로 생성)
        if os.path.exists(file_name):
            with pd.ExcelWriter(file_name, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Adjust column width
        wb = load_workbook(file_name)
        ws = wb[sheet_name]

        for col in ws.columns:
            max_length = 0
            col_letter = col[0].column_letter  # Get column letter
            for cell in col:
                try:
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))
                except:
                    pass
            ws.column_dimensions[col_letter].width = max_length + 2  # Adjust width with padding

        wb.save(file_name)

        print(f"Results saved to {file_name} with sheet name '{sheet_name}'.")  # 저장 완료 메시지 출력