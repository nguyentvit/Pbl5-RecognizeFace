import logging
from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    try:
        # Đọc file CSV
        df = pd.read_csv('dataRecognize.csv')
        # Lấy dữ liệu của hàng đầu tiên
        first_row = df.iloc[0].to_dict()
        return jsonify(first_row), 200
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return jsonify({"error": "Unable to read CSV file"}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, host='0.0.0.0', port=5000)
