import joblib
from joblib import dump

# Assuming `model` is your trained model or transformer

# Load the model
model = joblib.load('CSI-4900\model\models email content\stacking_model.joblib')
# dump(model, 'new_model_filename.joblib')
# Check if the version attribute exists
if hasattr(model, '__version__'):
    print(f"Model was trained with scikit-learn version: {model.__version__}")
else:
    print("Version information not found in the model.")
