from flask import Flask, request, jsonify
import joblib
from preprocessing import preprocessing_content
from flask_cors import CORS
import numpy as np
from eml_feature import extract_eml

app = Flask(__name__)
CORS(app)

# Load models
# model = joblib.load('model/new email models/stacking_model.joblib')
# vectorizer = joblib.load('model/new email models/vectorizer.joblib')
# scaler = joblib.load('model/new email models/scaler_model.joblib')

model = joblib.load('CSI-4900\\model\\new email models\\stacking_model.joblib')
vectorizer = joblib.load('CSI-4900\\model\\new email models\\vectorizer.joblib')
scaler = joblib.load('CSI-4900\\model\\new email models\\scaler_model.joblib')
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/analyze_email', methods=['POST'])
def analyze_email():
    # Check if a file is included in the request
    if 'eml_file' in request.files:
        file = request.files['eml_file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Extract content from the uploaded .eml file
        email_content = extract_eml(file)
        email_body = email_content.get("body", "")
        
        # Log the extracted email body
        logging.info(f'Extracted Email Body: {email_body}')
        
    else:
        # Get the email content from JSON
        email_content = request.json.get("email_content")
        email_content = email_content.lstrip()
        logging.info(f'Content:{email_content}')
        if not email_content:
            return jsonify({"error": "No email content provided"}), 400

        # Use the same function to preprocess text input as EML files
        email_info = preprocessing_content(email_content)
        email_body = email_info.get("body", "")

        # Log the preprocessed email body
        logging.info(f'Preprocessed Email Body: {email_body}')

    if not email_body:
        return jsonify({"error": "Email body is empty after processing"}), 400
    
    # Preprocess the email body
    X_text = vectorizer.transform([email_body])
    
    # Apply scaling
    X_scaled = scaler.transform(X_text.toarray())

    # Make a prediction
    prediction_proba_model = model.predict_proba(X_scaled)[0]

    # Calculate individual accuracies
    accuracy_model = prediction_proba_model[1]  # Probability of being spam

    # Determine the prediction label
    prediction_label = "Spam" if accuracy_model > 0.5 else "Not Spam"

    return jsonify({
        "Model_Accuracy": f"{max(prediction_proba_model[0], prediction_proba_model[1]) * 100:.2f}%",
        "Prediction": prediction_label
    })

if __name__ == '__main__':
    app.run(debug=True)
