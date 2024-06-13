import logging
from flask import Flask, jsonify
import pandas as pd
from datetime import datetime, timedelta
import requests
app = Flask(__name__)
url_server_web = 'http://192.168.1.37:5126/api/admin/Attendances'
@app.route('/')
def home():
    try:
        # Đọc file CSV
        df = pd.read_csv('dataRecognize.csv')
        # Lấy dữ liệu của hàng đầu tiên
        first_row = df.iloc[0].to_dict()
        file_time = datetime.fromtimestamp(int(first_row['Time']))
        current_time = datetime.now() - timedelta(seconds=5)
        if file_time > current_time:
            data = {"status": first_row["Status"], "userId": first_row["Id"], "pathImg": first_row["Time"]}
            response = requests.post(url_server_web, json=data)
            if response.status_code == 201:
                return jsonify(data), 200
            else:
                data_err = {"Status": False, "id": False, "Time": False, "personName": False,
                        "StatusRecognized": False, "Img_Path": False, "post": False}
                print(response)
                return jsonify(data_err), 500
        else:
            data = {"Status": False, "id": False, "Time": False, "personName": False,
                    "StatusRecognized": False, "Img_Path": False}
            return jsonify(data), 200
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return jsonify({"error": "Unable to read CSV file"}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, host='0.0.0.0', port=5000)
