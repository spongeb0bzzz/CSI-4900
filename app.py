from flask import Flask, request, jsonify, send_file
import joblib
from preprocessing import preprocessing_content
from flask_cors import CORS
import numpy as np
from eml_feature import extract_eml,extract_eml_body
import logging
from get_URL_features import extract_links,extract_features
import pandas as pd
from get_scores import get_average_similarity, get_result_from_database  # database score and CBR
from concurrent.futures import ThreadPoolExecutor
from lime.lime_text import LimeTextExplainer
import os

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load('model/new email models/stacking_model.joblib')
    vectorizer = joblib.load('model/new email models/vectorizer.joblib')
    scaler = joblib.load('model/new email models/scaler_model.joblib')
    model_url = joblib.load('model/models url/stacking_model(2).joblib')
    scaler_url = joblib.load('model/models url/scaler(2).joblib')
    print("Loaded models using first path.")
except FileNotFoundError:
    # Attempt to load models with the second path
    try:
        model = joblib.load('CSI-4900\\model\\new email models\\stacking_model.joblib')
        vectorizer = joblib.load('CSI-4900\\model\\new email models\\vectorizer.joblib')
        scaler = joblib.load('CSI-4900\\model\\new email models\\scaler_model.joblib')
        model_url = joblib.load('CSI-4900\\model\\models url\\stacking_model(2).joblib')
        scaler_url = joblib.load('CSI-4900\\model\\models url\\scaler(2).joblib')
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
    ## Link Prediction (old version)
    ############################################################################################################################################################
    
    predictions_url = []
    # Process URL predictions if there are links
    # if links:
    #     for url in links:


    #         db_results = get_result_from_database(url)  

    #         #Get the similarity score using CBR
    #         cbr_score = get_average_similarity(url) 

    #         if db_results is not None:
    #             if db_results == 1 :
    #                 # If any database score is phishing, return phishing and stop further checks
    #                 predictions_url.append({
    #                     'url': url,
    #                     'prediction_label': "Spam",
    #                     'accuracy_model': "100.00%",  # Database result is conclusive
    #                     'spam_rate': 1.0,
    #                     'db_score': 1,
    #                     'cbr': cbr_score
    #                 })
    #                 continue
    #             elif db_results == 0:
    #                 # If any database score is benign, return safe and stop further checks
    #                 predictions_url.append({
    #                     'url': url,
    #                     'prediction_label': "Not Spam",
    #                     'accuracy_model': "100.00%",  # Database result is conclusive
    #                     'spam_rate': 0.0,
    #                     'db_score': 0,
    #                     'cbr': cbr_score
    #                 })
    #                 continue

    #         else:
            




    #             features = extract_features(url)
    #             features_df = pd.DataFrame([features])
    #             X_scaled_url = scaler_url.transform(features_df)
    #             prediction_proba_model_url = model_url.predict_proba(X_scaled_url)[0]
    #             accuracy_model_url = prediction_proba_model_url[1]
    #             prediction_label_url = "Spam" if accuracy_model_url > 0.5 else "Not Spam"
    #             predictions_url.append({
    #                 'url': url,
    #                 'prediction_label': prediction_label_url,
    #                 'accuracy_model': f"{max(prediction_proba_model_url[0], prediction_proba_model_url[1]) * 100:.2f}%",
    #                 'spam_rate': accuracy_model_url,
    #                 'db_score':db_results,
    #                 'cbr': cbr_score
    #             })

    # logging.info(f'PD: {predictions_url}')


    ############################################################################################################################################################
    ## Parallel running for link prediction 
    ############################################################################################################################################################
    
    #define three functions for url analysis
    def get_db_score(url):
        
        return get_result_from_database(url)
    
    def get_cbr_score(url):
       
        return get_average_similarity(url)

    def get_url_prediction(url):
        
        features = extract_features(url)
        features_df = pd.DataFrame([features])
        X_scaled_url = scaler_url.transform(features_df)
        prediction_proba_model_url = model_url.predict_proba(X_scaled_url)[0]
        accuracy_model_url = prediction_proba_model_url[1]
        prediction_label_url = "Spam" if accuracy_model_url > 0.5 else "Not Spam"
        return {
            'prediction_label': prediction_label_url,
            'accuracy_model': f"{max(prediction_proba_model_url[0], prediction_proba_model_url[1]) * 100:.2f}%",
            'spam_rate': accuracy_model_url
        }
    
    def process_url_parallel(url):
        #run code parallel
        with ThreadPoolExecutor() as executor:
            db_future = executor.submit(get_db_score, url)
            cbr_future = executor.submit(get_cbr_score, url)
            url_prediction_future = executor.submit(get_url_prediction, url)
            
            db_results = db_future.result()
            cbr_score = cbr_future.result()
            url_prediction = url_prediction_future.result()

        # Combine results based on priority
        if db_results is not None:
            if db_results == 1:
                return {
                    'url': url,
                    'prediction_label': "Spam",
                    'accuracy_model': "100.00%",
                    'spam_rate': 1.0,
                    'db_score': 1,
                    'cbr': cbr_score
                }
            elif db_results == 0:
                return {
                    'url': url,
                    'prediction_label': "Not Spam",
                    'accuracy_model': "100.00%",
                    'spam_rate': 0.0,
                    'db_score': 0,
                    'cbr': cbr_score
                }

        # Fallback to model prediction if no db_results
        return {
            'url': url,
            'prediction_label': url_prediction['prediction_label'],
            'accuracy_model': url_prediction['accuracy_model'],
            'spam_rate': url_prediction['spam_rate'],
            'db_score': db_results,
            'cbr': cbr_score
        }
    
    if links:
        with ThreadPoolExecutor() as executor:
            predictions_url = list(executor.map(process_url_parallel, links))

    ############################################################################################################################################################
    ## Content Prediction
    ############################################################################################################################################################

    X_text = vectorizer.transform([email_body])
    X_scaled = scaler.transform(X_text.toarray())
    prediction_proba_model = model.predict_proba(X_scaled)[0]
    accuracy_model = prediction_proba_model[1]
    prediction_label = "Spam" if accuracy_model > 0.5 else "Not Spam"

    ############################################################################################################################################################
    ## LIME 
    ############################################################################################################################################################

    def predict_proba_with_reshape(X):
    

        # Ensure X is a list of strings (raw email content)
        if isinstance(X, np.ndarray) or isinstance(X, list):
            # Convert raw email content into numerical features using vectorizer
            X = vectorizer.transform(X)  # Outputs a sparse matrix

        # Convert sparse matrix to dense if necessary
        if not isinstance(X, np.ndarray):
            X = X.toarray()  # Ensure compatibility with scaler and model

        # Reshape if input is 1D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Scale the data (if scaler is used)
        X = scaler.transform(X)

        # Return prediction probabilities
        return model.predict_proba(X)
    
    explainer = LimeTextExplainer(class_names=["Not Spam", "Spam"])
    explanation = explainer.explain_instance(
        email_body,  # Raw email text
        predict_proba_with_reshape,  # Prediction function
        num_features=10  # Number of features to highlight
    )
    explanation_file = "lime_explanation.html"
    explanation.save_to_file(explanation_file)

    ############################################################################################################################################################
    ## Combine Result 
    ############################################################################################################################################################

    # Final output calculation

    if predictions_url:
    # Check if any URL in the list has a db_score of 1 or 0
        db_scores = [item.get('db_score', -1) for item in predictions_url]
        
        if 1 in db_scores:
            # If any db_score is 1, final label is Spam with 100% accuracy
            final_label = "Spam"
            final_accuracy = 1.0
        elif 0 in db_scores:
            # If any db_score is 0, final label is Not Spam with 100% accuracy
            final_label = "Not Spam"
            final_accuracy = 1.0
        else:
            # Extract values for accuracies, cbr, and db_score
            accuracies = [float(item['spam_rate']) for item in predictions_url]
            cbr_final = [float(item['cbr']) for item in predictions_url]
            db_final = [float(item['db_score']) for item in predictions_url if item.get('db_score') is not None]

            # Calculate averages for the respective lists
            average_db_list = sum(db_final) / len(db_final) if db_final else None
            average_cbr_list = sum(cbr_final) / len(cbr_final) if cbr_final else 0
            average_accuracy_list = sum(accuracies) / len(accuracies) if accuracies else 0

            # Final score calculation based on the presence of db scores
            if average_db_list is not None:
                final_output = (
                    0.6 * average_accuracy_list +
                    0.2 * accuracy_model +
                    0.1 * average_cbr_list +
                    0.1 * average_db_list
                )
            else:
                # Adjust formula if db scores are missing
                final_output = (
                    0.7 * average_accuracy_list +
                    0.2 * accuracy_model +
                    0.1 * average_cbr_list
                )

            # Determine final label and accuracy
            final_label = "Spam" if final_output > 0.5 else "Not Spam"
            final_accuracy = final_output if final_label == 'Spam' else 1 - final_output
    else:
        # Fallback if predictions_url is empty
        final_label = "Spam" if accuracy_model > 0.5 else "Not Spam"
        final_accuracy = accuracy_model if final_label == 'Spam' else 1 - accuracy_model



    # logging.info(f'DB: {db_final}, CBR: {cbr_final}')

    ############################################################################################################################################################
    ## Return Data
    ############################################################################################################################################################

    response_data = {
        "Model_Accuracy": f"{max(prediction_proba_model[0], prediction_proba_model[1]) * 100:.2f}%",
        "Prediction": prediction_label,
        "links": predictions_url,
        "output": f"{final_accuracy * 100:.2f}%",
        "OutputLabel": final_label,
        "LIME_Explanation_URL": f"http://127.0.0.1:5000/lime_explanation"
    }
    logging.info(f'Response Data: {response_data}')
    return jsonify(response_data)

@app.route('/lime_explanation', methods=['GET'])
def lime_explanation():
    return send_file("lime_explanation.html", mimetype="text/html")

if __name__ == '__main__':
    app.run(debug=True)
