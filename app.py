from flask import Flask, request, jsonify
import joblib
from preprocessing import preprocessing_content
from flask_cors import CORS
import numpy as np
from eml_feature import extract_eml,extract_eml_body
import logging
from get_URL_features import extract_links,extract_features
import pandas as pd

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load('model/new email models/stacking_model.joblib')
    vectorizer = joblib.load('model/new email models/vectorizer.joblib')
    scaler = joblib.load('model/new email models/scaler_model.joblib')
    model_url = joblib.load('model/models url/stacking_model(1).joblib')
    scaler_url = joblib.load('model/models url/scaler(1).joblib')
    print("Loaded models using first path.")
except FileNotFoundError:
    # Attempt to load models with the second path
    try:
        model = joblib.load('CSI-4900\\model\\new email models\\stacking_model.joblib')
        vectorizer = joblib.load('CSI-4900\\model\\new email models\\vectorizer.joblib')
        scaler = joblib.load('CSI-4900\\model\\new email models\\scaler_model.joblib')
        model_url = joblib.load('CSI-4900\\model\\models url\\stacking_model(1).joblib')
        scaler_url = joblib.load('CSI-4900\\model\\models url\\scaler(1).joblib')
        print("Loaded models using second path.")
    except FileNotFoundError:
        print("Error: Unable to load models from either path.")


# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/analyze_email', methods=['POST'])
def analyze_email():

    ############################################################################################################################################################
    ## GET Content from EML file
    ############################################################################################################################################################

    # Initialize email_body to prevent UnboundLocalError
    email_body = None

    # Check if a file is included in the request
    if 'eml_file' in request.files:
        logging.info(f'Working with eml file')
        file = request.files['eml_file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        # Read and decode content from the file directly
        try:
            email_content = extract_eml_body(file)
            # Check if email_content is a dictionary
            if isinstance(email_content, dict):
                email_body = email_content.get("body_plain", "")
            else:
                email_body = email_content  # Assume it's a string if not a dictionary
            email_info = preprocessing_content(email_body)
        except Exception as e:
            logging.error(f"Error extracting email content: {e}")
            return jsonify({"error": "Failed to process the email file"}), 400
    
    ############################################################################################################################################################
    ## GET Content from Pasting
    ############################################################################################################################################################

    else:
        # Get the email content from JSON
        email_content = request.json.get("email_content", "").strip()
        if not email_content:
            return jsonify({"error": "No email content provided"}), 400
        email_info = preprocessing_content(email_content)
        email_body = email_info.get("body", "")

    if not email_body:
        return jsonify({"error": "Email body is empty after processing"}), 400

    ############################################################################################################################################################
    ## GET THE LINKS From Email body
    ############################################################################################################################################################

    links = extract_links(email_body)
    logging.info(f'Extracted Email Body: {email_body}')
    logging.info(f'Contain links: {links}')
    
    ############################################################################################################################################################
    ## Link Prediction
    ############################################################################################################################################################
    
    predictions_url = []
    # Process URL predictions if there are links
    if links:
        for url in links:
            features = extract_features(url)
            features_df = pd.DataFrame([features])
            X_scaled_url = scaler_url.transform(features_df)
            prediction_proba_model_url = model_url.predict_proba(X_scaled_url)[0]
            accuracy_model_url = prediction_proba_model_url[1]
            prediction_label_url = "Spam" if accuracy_model_url > 0.5 else "Not Spam"
            predictions_url.append({
                'url': url,
                'prediction_label': prediction_label_url,
                'accuracy_model': f"{max(prediction_proba_model_url[0], prediction_proba_model_url[1]) * 100:.2f}%",
                'spam_rate': accuracy_model_url
            })

    ############################################################################################################################################################
    ## Content Prediction
    ############################################################################################################################################################

    X_text = vectorizer.transform([email_body])
    X_scaled = scaler.transform(X_text.toarray())
    prediction_proba_model = model.predict_proba(X_scaled)[0]
    accuracy_model = prediction_proba_model[1]
    prediction_label = "Spam" if accuracy_model > 0.5 else "Not Spam"

    ############################################################################################################################################################
    ## Combine Result 
    ############################################################################################################################################################

    # Final output calculation
    accuracies = [float(item['spam_rate']) for item in predictions_url]
    average_accuracy_list = sum(accuracies) / len(accuracies) if accuracies else 0
    final_output = (0.7 * average_accuracy_list + 0.3 * accuracy_model) if average_accuracy_list else accuracy_model
    final_label = "Spam" if final_output > 0.5 else "Not Spam"
    final_accuracy = final_output if final_label == 'Spam' else 1 - final_output

    ############################################################################################################################################################
    ## Return Data
    ############################################################################################################################################################

    response_data = {
        "Model_Accuracy": f"{max(prediction_proba_model[0], prediction_proba_model[1]) * 100:.2f}%",
        "Prediction": prediction_label,
        "links": predictions_url,
        "output": f"{final_accuracy * 100:.2f}%",
        "OutputLabel": final_label
    }
    logging.info(f'Response Data: {response_data}')
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
