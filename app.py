from flask import Flask, request, jsonify
import joblib
from prepossessing import prepossessing_content
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# 加载预训练模型
# phishing_model = joblib.load("path/to/your/model.joblib")

@app.route('/analyze_email', methods=['POST'])
def analyze_email():
    email_content = request.json.get("email_content")
    email_content = prepossessing_content(email_content)


    # prediction = phishing_model.predict([email_content["body"]])[0]
    # result = "Phishing" if prediction == 1 else "Legit"


    return jsonify({"result": email_content["body"]})

if __name__ == '__main__':
    app.run(debug=True)
