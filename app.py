from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

logisticRegression = joblib.load('linear_regression_model.pkl')
randomForest = joblib.load('random_forest_classifier.pkl')
supportVectorMachine = joblib.load('support_vector_machine.pkl')

data = pd.read_csv('heart.csv')

binary_features = ['fbs', 'restecg']
continuous_features = ['age', 'trtbps', 'chol', 'thalachh', 'cp', 'slp']
features = binary_features + continuous_features

X = data[features]
y = data['output'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)
scaler = StandardScaler()
X_train[continuous_features] = scaler.fit_transform(X_train[continuous_features])
X_test[continuous_features] = scaler.transform(X_test[continuous_features])

models = {
    'LogisticRegression': logisticRegression,
    'RandomForest': randomForest,
    'SVM': supportVectorMachine
}

metrics = {}

# Treinar e avaliar os modelos
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

def predict_hearthattack_risk(input_data, model_name):
    model = models[model_name]
    input_df = pd.DataFrame([input_data])
    
    for col in binary_features + continuous_features:
        if col not in input_df.columns:
            raise ValueError(f"Falta a coluna necessária: {col}")
    
    missing_cols = set(X_train.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    
    input_df = input_df[X_train.columns] 
    input_df[continuous_features] = scaler.transform(input_df[continuous_features])
    
    # Verificação de depuração
    print("Dados de entrada após transformação:")
    print(input_df)
    
    if input_df.isnull().values.any():
        raise ValueError("O DataFrame de entrada contém NaN após a transformação.")
    
    probability = model.predict_proba(input_df)[:, 1]
    return probability[0]

@app.route('/')
def home():
    return render_template('index.html', features=binary_features + continuous_features)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            model_name = request.form['model']
            input_data = {
                'age': float(request.form['age']),
                'cp': int(request.form['cp']),
                'trtbps': float(request.form['trtbps']),
                'chol': float(request.form['chol']),
                'fbs': int(request.form['fbs']),
                'restecg': int(request.form['restecg']),
                'thalachh': float(request.form['thalachh']),
                'slp': int(request.form['slp'])
            }
            probability = predict_hearthattack_risk(input_data, model_name)
            return jsonify(probability=probability)
        except KeyError as e:
            error_message = f"Faltando campo obrigatório: {str(e)}"
            print(error_message)
            return jsonify(error=error_message), 400
        except ValueError as e:
            error_message = str(e)
            print(error_message)
            return jsonify(error=error_message), 400
    else:
        return render_template('index.html', features=binary_features + continuous_features)

@app.route('/metrics', methods=['GET'])
def get_metrics():
    model_name = request.args.get('model')
    if model_name:
        return jsonify(metrics.get(model_name, {}))
    else:
        return jsonify({"error": "Model not specified"}), 400

if __name__ == '__main__':
    app.run(debug=True)
