from flask import Flask, request, jsonify
import joblib
from prepossessing import prepossessing_content
from flask_cors import CORS
import numpy as np
from eml_feature import extract_eml
# import warnings
# from sklearn.exceptions import InconsistentVersionWarning

# # Suppress InconsistentVersionWarning
# warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)
CORS(app)
#Load

# phishing_model = joblib.load("CSI-4900\model\decision_tree_model_content.joblib")

model = joblib.load('CSI-4900\model\models email content\stacking_model (1).joblib')
# model = joblib.load('CSI-4900\model\models email content\svm_model (1).joblib')
vectorizer = joblib.load('CSI-4900\\model\\vectorizer.joblib')

@app.route('/analyze_email', methods=['POST'])
def analyze_email():
    # Check if a file is included in the request
    if 'eml_file' in request.files:
        file = request.files['eml_file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Extract content from the uploaded .eml file
        email_content = extract_eml(file)
        email_body = email_content.get("body_plain", "")
    else:
        # Get the email content from JSON
        email_content = request.json.get("email_content")
        if not email_content:
            return jsonify({"error": "No email content provided"}), 400

        # Preprocess the email content
        email_content = prepossessing_content(email_content)
        email_body = email_content.get("body", "")

    if not email_body:
        return jsonify({"error": "Email body is empty after processing"}), 400
    
    # Preprocess the email body
    X_text = vectorizer.transform([email_body])

    # Make a prediction
    prediction_proba_model = model.predict_proba(X_text)[0]

    # Output the model prediction probabilities
    # print(f"Model Prediction probabilities: {prediction_proba_model}")

    # Define a prediction function for SHAP
    def model_predict(X):
        return model.predict_proba(X)

    # Calculate individual accuracies
    accuracy_model = prediction_proba_model[1]  # Probability of being spam

    # Determine the prediction label
    prediction_label = "Spam" if accuracy_model > 0.5 else "Not Spam"


    return jsonify({
    "Model_Accuracy": f"{max(prediction_proba_model[0], prediction_proba_model[1]) * 100:.2f}%",
    "Prediction": prediction_label})

if __name__ == '__main__':
    app.run(debug=True)
