import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import random
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Title
st.title("Parkinson's Disease Detection Using Voice Attributes")

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('parkinsons_1.csv')
    X = data.drop(['name', 'status'], axis=1, errors='ignore')
    y = data['status']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    return X_train, X_test, y_train, y_test

# Build model
def build_model(model_type, input_shape):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=input_shape))
        model.add(LSTM(100, activation='relu'))
        model.add(Dropout(0.1))
    elif model_type == 'GRU':
        model.add(GRU(100, activation='relu', return_sequences=True, input_shape=input_shape))
        model.add(GRU(100, activation='relu'))
        model.add(Dropout(0.1))
    elif model_type == 'Hybrid LSTM-GRU':
        model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=input_shape))
        model.add(LSTM(100, activation='relu', return_sequences=True))
        model.add(Dropout(0.1))
        model.add(GRU(256, return_sequences=True))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Train and evaluate
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    class StreamlitCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            st.write(f"Epoch {epoch + 1}/50 - Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[StreamlitCallback()], verbose=0)

    # # Evaluate on test data
    # test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    # st.write(f"✅ Final Test Loss: {test_loss:.4f}")
    # st.write(f"✅ Final Test Accuracy: {test_accuracy:.4f}")

    y_pred = (model.predict(X_test) > 0.5).astype(int).reshape(-1)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return acc, prec, rec, f1, cm, report

# State for results
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

# Load data
X_train, X_test, y_train, y_test = load_data()
feature_size = X_train.shape[2]

# Model selection
st.sidebar.title("Model Selection")
model_type = st.sidebar.selectbox("Choose Model", ['LSTM', 'GRU', 'Hybrid LSTM-GRU'])

if st.sidebar.button("Train Model"):
    st.write(f"Training {model_type} Model...")
    model = build_model(model_type, (None, feature_size))
    acc, prec, rec, f1, cm, report = train_and_evaluate(model, X_train, y_train, X_test, y_test)
    st.session_state.model_results[model_type] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'Confusion Matrix': cm,
        'Report': report
    }
    st.success(f"{model_type} Model Trained Successfully!")

# Compare results
if st.sidebar.button("Compare Models"):
    if len(st.session_state.model_results) < 3:
        st.warning("Please train all three models (LSTM, GRU, Hybrid LSTM-GRU) to compare.")
    else:
        results = st.session_state.model_results
        df = pd.DataFrame({
            'Model': ['LSTM', 'GRU', 'Hybrid LSTM-GRU'],
            'Accuracy': [results['LSTM']['Accuracy'], results['GRU']['Accuracy'], results['Hybrid LSTM-GRU']['Accuracy']],
            'Precision': [results['LSTM']['Precision'], results['GRU']['Precision'], results['Hybrid LSTM-GRU']['Precision']],
            'Recall': [results['LSTM']['Recall'], results['GRU']['Recall'], results['Hybrid LSTM-GRU']['Recall']],
            'F1-Score': [results['LSTM']['F1-Score'], results['GRU']['F1-Score'], results['Hybrid LSTM-GRU']['F1-Score']]
        }).set_index('Model')

        st.dataframe(df)
        # Line plot for performance comparison
        st.write("### Performance Line Plot")

        fig, ax = plt.subplots(figsize=(8, 4))
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            ax.plot(df.index, df[metric], marker='o', label=metric)

        ax.set_ylim(0.85, 1.0)  # Zoom in if values are high and close
        ax.set_ylabel("Score")
        ax.set_title("Model Metrics Comparison")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
