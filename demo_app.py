from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from keras.models import load_model
from joblib import load
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load tất cả các models
models = {}

# Load ANN
try:
    models['ann'] = load_model('./results/model/ann/ann_best.h5')
    print("✓ ANN model loaded")
except Exception as e:
    print(f"✗ ANN: {e}")

# Load KNN
try:
    models['knn'] = load('./results/model/knn/knn_best.pkl')
    print("✓ KNN model loaded")
except Exception as e:
    print(f"✗ KNN: {e}")

# Load Naive Bayes
try:
    models['naive_bayes'] = load('./results/model/naive_bayes/nb_best.pkl')
    print("✓ Naive Bayes model loaded")
except Exception as e:
    print(f"✗ Naive Bayes: {e}")

# Load Random Forest
try:
    models['random_forest'] = load('./results/model/random_forest/rf_best.pkl')
    print("✓ Random Forest model loaded")
except Exception as e:
    print(f"✗ Random Forest: {e}")

# Load K-means
try:
    models['k_means'] = load('./results/model/k_means/kmeans_best.pkl')
    print("✓ K-means model loaded")
except Exception as e:
    print(f"✗ K-means: {e}")

# Tên các loài penguin
species_names = ["Adelie", "Chinstrap", "Gentoo"]

# Các features (7 features - không có Island)
feature_names = [
    "Culmen Length (mm)",
    "Culmen Depth (mm)", 
    "Flipper Length (mm)",
    "Body Mass (g)",
    "Delta 15 N",
    "Delta 13 C",
    "Sex (0=Female, 1=Male)"
]

@app.route('/')
def index():
    return render_template('index.html', 
                         models=list(models.keys()),
                         features=feature_names,
                         species=species_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_name = data['model']
        features = data['features']
        
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} không tồn tại'}), 400
        
        # Chuyển đổi input thành numpy array
        X = np.array(features).reshape(1, -1)
        
        # Dự đoán
        model = models[model_name]
        prediction = model.predict(X)
        
        # Xử lý kết quả dựa trên loại model
        if model_name == 'ann':
            # Deep learning models return probabilities
            probabilities = prediction[0].tolist()
            predicted_class = int(np.argmax(prediction))
        elif model_name == 'k_means':
            # K-means is unsupervised, returns cluster labels
            pred_value = prediction.flat[0] if hasattr(prediction, 'flat') else prediction[0]
            predicted_class = int(pred_value) if not hasattr(pred_value, 'item') else pred_value.item()
            probabilities = [0.0, 0.0, 0.0]
            probabilities[predicted_class] = 1.0
        elif model_name in ['knn', 'random_forest']:
            # KNN and Random Forest were trained with one-hot encoded labels
            # Use argmax to get the class index
            predicted_class = int(np.argmax(prediction[0]))
            probabilities = [0.0, 0.0, 0.0]
            probabilities[predicted_class] = 1.0
        else:
            # Other sklearn models
            pred_value = prediction.flat[0] if hasattr(prediction, 'flat') else prediction[0]
            predicted_class = int(pred_value) if not hasattr(pred_value, 'item') else pred_value.item()
            # Try to get probability
            if hasattr(model, 'predict_proba'):
                prob_array = model.predict_proba(X)
                # Convert properly - ensure we get Python floats
                prob_np = np.array(prob_array)
                # Extract first row as 1D array
                probs_1d = prob_np[0]
                # Convert to Python list of Python floats with explicit handling
                probabilities = []
                for i in range(len(probs_1d)):
                    val = probs_1d[i]
                    # Use item() to convert numpy scalar to Python scalar
                    if hasattr(val, 'item'):
                        probabilities.append(val.item())
                    else:
                        probabilities.append(float(val))
            else:
                # Create one-hot like probabilities
                probabilities = [0.0, 0.0, 0.0]
                probabilities[predicted_class] = 1.0
        
        result = {
            'model': model_name,
            'predicted_class': predicted_class,
            'predicted_species': species_names[predicted_class],
            'probabilities': {
                species_names[i]: round(prob * 100, 2)
                for i, prob in enumerate(probabilities)
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_all', methods=['POST'])
def predict_all():
    """Dự đoán với tất cả các models cùng lúc"""
    try:
        data = request.json
        features = data['features']
        
        X = np.array(features).reshape(1, -1)
        results = []
        
        for model_name, model in models.items():
            try:
                prediction = model.predict(X)
                
                if model_name == 'ann':
                    probabilities = prediction[0].tolist()
                    predicted_class = int(np.argmax(prediction))
                elif model_name == 'k_means':
                    pred_value = prediction.flat[0] if hasattr(prediction, 'flat') else prediction[0]
                    predicted_class = int(pred_value) if not hasattr(pred_value, 'item') else pred_value.item()
                    probabilities = [0.0, 0.0, 0.0]
                    probabilities[predicted_class] = 1.0
                elif model_name in ['knn', 'random_forest']:
                    # KNN and Random Forest were trained with one-hot encoded labels
                    predicted_class = int(np.argmax(prediction[0]))
                    probabilities = [0.0, 0.0, 0.0]
                    probabilities[predicted_class] = 1.0
                else:
                    # Other sklearn models
                    pred_value = prediction.flat[0] if hasattr(prediction, 'flat') else prediction[0]
                    predicted_class = int(pred_value) if not hasattr(pred_value, 'item') else pred_value.item()
                    if hasattr(model, 'predict_proba'):
                        prob_array = model.predict_proba(X)
                        # Convert properly - ensure we get Python floats
                        prob_np = np.array(prob_array)
                        # Extract first row as 1D array
                        probs_1d = prob_np[0]
                        # Convert to Python list of Python floats with explicit handling
                        probabilities = []
                        for i in range(len(probs_1d)):
                            val = probs_1d[i]
                            # Use item() to convert numpy scalar to Python scalar
                            if hasattr(val, 'item'):
                                probabilities.append(val.item())
                            else:
                                probabilities.append(float(val))
                    else:
                        probabilities = [0.0, 0.0, 0.0]
                        probabilities[predicted_class] = 1.0
                
                results.append({
                    'model': model_name,
                    'predicted_class': predicted_class,
                    'predicted_species': species_names[predicted_class],
                    'probabilities': {
                        species_names[i]: round(prob * 100, 2)
                        for i, prob in enumerate(probabilities)
                    }
                })
            except Exception as e:
                results.append({
                    'model': model_name,
                    'error': str(e)
                })
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models_info')
def models_info():
    """Trả về thông tin về các models đã load"""
    info = {}
    for name, model in models.items():
        info[name] = {
            'loaded': True,
            'type': type(model).__name__
        }
    return jsonify(info)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("PENGUIN SPECIES CLASSIFICATION DEMO")
    print("="*50)
    print(f"Đã load {len(models)} models")
    print("\nMở trình duyệt và truy cập: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
