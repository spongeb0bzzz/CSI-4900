from flask import Flask, request, jsonify
import joblib
from preprocessing import preprocessing_content
from flask_cors import CORS
import numpy as np
from eml_feature import extract_eml
import logging
from get_URL_features import extract_links,extract_features
import pandas as pd

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load('model/new email models/stacking_model.joblib')
    vectorizer = joblib.load('model/new email models/vectorizer.joblib')
    scaler = joblib.load('model/new email models/scaler_model.joblib')
    model_url = joblib.load('model/models url/stacking_model.joblib')
    scaler_url = joblib.load('model/models url/scaler.joblib')
    print("Loaded models using first path.")
except FileNotFoundError:
    # Attempt to load models with the second path
    try:
        model = joblib.load('CSI-4900\\model\\new email models\\stacking_model.joblib')
        vectorizer = joblib.load('CSI-4900\\model\\new email models\\vectorizer.joblib')
        scaler = joblib.load('CSI-4900\\model\\new email models\\scaler_model.joblib')
        model_url = joblib.load('CSI-4900\\model\\models url\\stacking_model.joblib')
        scaler_url = joblib.load('CSI-4900\\model\\models url\\scaler.joblib')
        print("Loaded models using second path.")
    except FileNotFoundError:
        print("Error: Unable to load models from either path.")


# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/analyze_email', methods=['POST'])
def analyze_email():
    logging.info(f'Request files: {request.files}')
    # Check if a file is included in the request
    if 'eml_file' in request.files:
        logging.info(f'Working with eml file')
        file = request.files['eml_file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        # Read and decode content from the file directly
        raw_content = file.read()
        email_content = raw_content.decode('utf-8', errors='replace') if isinstance(raw_content, bytes) else raw_content
        # Extract content from the uploaded .eml file
        email_content = extract_eml(email_content)

        # logging.info(f'Email content: {email_content}')
        
        email_body = email_content.get("body_plain", "")
        email_info = preprocessing_content(email_body)
        email_body = email_info.get("body", "")
        
        
        # Log the extracted email body
        logging.info(f'Extracted Email Body: {email_body}')
        
    else:
        logging.info(f'Working with pasting content')
        # Get the email content from JSON
        email_content = request.json.get("email_content")
        email_content = email_content.lstrip()
        # logging.info(f'Content:{email_content}')
        if not email_content:
            return jsonify({"error": "No email content provided"}), 400

        # Use the same function to preprocess text input as EML files
        email_info = preprocessing_content(email_content)
        email_body = email_info.get("body", "")
        logging.info(f'Email info: {email_info}')
        # Log the preprocessed email body
        logging.info(f'Preprocessed Email Body: {email_body}')

    if not email_body:
        return jsonify({"error": "Email body is empty after processing"}), 400
    
    links = extract_links(email_body)
    logging.info(f'Contain links:{links}')


    if links is not None:
        predictions_url = []
        
        for url in links:
            features = extract_features(url)
            features_df = pd.DataFrame([features])  # Convert to DataFrame with one row


            X_scaled_url = scaler_url.transform(features_df)

            prediction_proba_model_url = model_url.predict_proba(X_scaled_url)[0]

            # Probability of being spam
            accuracy_model_url = prediction_proba_model_url[1]  

            # Determine the prediction label
            prediction_label_url = "Spam" if accuracy_model_url > 0.5 else "Not Spam"

            # Append the result for each URL
            predictions_url.append({     
                'url': url,
                'prediction_label': prediction_label_url,
                'accuracy_model': accuracy_model_url
            })

    logging.info(f'Url prediction: {predictions_url}')


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
