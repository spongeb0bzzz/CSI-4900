from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# 加载预训练模型
phishing_model = joblib.load("path/to/your/model.joblib")

@app.route('/analyze_email', methods=['POST'])
def analyze_email():
    email_content = request.json.get("email_content")
    prediction = phishing_model.predict([email_content])[0]
    result = "Phishing" if prediction == 1 else "Legit"
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
