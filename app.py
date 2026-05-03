from flask import Flask, render_template, request, redirect, session
import pandas as pd
import sqlite3
import re
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from textblob import TextBlob

app = Flask(__name__)
app.secret_key = "fraud_secret_key_2026"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

TARGET = "FraudFound_P"

# Globals
label_encoders = {}
scaler = None
model = None
model_columns = []

accuracy = precision_val = recall_val = f1_val = 0
prediction_result = "No prediction yet"
prediction_label = None
sentiment_result = None

# ================= DATABASE =================
def init_db():
    with sqlite3.connect("database.db") as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL
            )
        """)
init_db()

# ================= PASSWORD =================
def is_valid_password(password):
    return re.match(r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d).{8,}$', password)

# ================= LOAD DATA =================
df = pd.read_csv("insurance_claims.csv")

# ================= SENTIMENT =================
def analyze_sentiment(text):
    if not text.strip():
        return {"label": "N/A", "risk_flag": False}

    polarity = TextBlob(text).sentiment.polarity
    label = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

    suspicious = ["no proof", "not sure", "forgot", "no witness"]
    risk_flag = any(w in text.lower() for w in suspicious)

    return {"label": label, "risk_flag": risk_flag}

# ================= PREPROCESS =================
def preprocess_data(dataframe):
    global label_encoders

    df_copy = dataframe.copy()

    # Handle missing values
    df_copy = df_copy.fillna("Unknown")

    label_encoders = {}

    for col in df_copy.columns:
        if df_copy[col].dtype == "object":
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
            label_encoders[col] = le

    return df_copy

# ================= TRAIN =================
def train_model():
    global model, scaler, model_columns
    global accuracy, precision_val, recall_val, f1_val

    df_clean = df.copy()
    df_clean = df_clean.fillna("Unknown")

    y = df_clean[TARGET]
    X = df_clean.drop(columns=[TARGET])

    # Encode features only
    X = preprocess_data(X)

    model_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42
    )

    # Apply SMOTE safely
    try:
        smote = SMOTE(sampling_strategy=0.7, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    except Exception as e:
        print("⚠️ SMOTE skipped:", e)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=15,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    threshold = 0.65
    preds = (probs >= threshold).astype(int)

    accuracy = accuracy_score(y_test, preds)
    precision_val = precision_score(y_test, preds, zero_division=0)
    recall_val = recall_score(y_test, preds, zero_division=0)
    f1_val = f1_score(y_test, preds, zero_division=0)

    print("✅ Model trained successfully")

train_model()

# ================= HELPERS =================
def metrics():
    return dict(
        accuracy=round(accuracy * 100, 2),
        precision=round(precision_val * 100, 2),
        recall=round(recall_val * 100, 2),
        f1=round(f1_val * 100, 2)
    )

def login_required(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    return wrapper

# ================= ROUTES =================

@app.route('/')
def home():
    return redirect('/login')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == "POST":
        u = request.form['username']
        p = request.form['password']

        with sqlite3.connect("database.db") as conn:
            user = conn.execute("SELECT * FROM users WHERE username=? AND password=?", (u,p)).fetchone()

        if user:
            session['user'] = u
            return redirect('/dashboard')

    return render_template("login.html")

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == "POST":
        u = request.form['username']
        p = request.form['password']

        if not is_valid_password(p):
            return "Weak Password"

        with sqlite3.connect("database.db") as conn:
            conn.execute("INSERT INTO users VALUES (?,?)", (u,p))

        return redirect('/login')

    return render_template("signup.html")

@app.route('/dashboard')
@login_required
def dashboard():
    fraud_counts = df[TARGET].value_counts().to_dict()

    return render_template(
        "dashboard.html",
        fraud_counts=json.dumps(fraud_counts),
        **metrics()
    )

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    global df
    if request.method == "POST":
        file = request.files['file']
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            df = pd.read_csv(path)
            train_model()
            return redirect('/dashboard')

    return render_template("upload.html")

@app.route('/predict_page')
@login_required
def predict_page():
    return render_template("form.html", columns=model_columns)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    global prediction_result, prediction_label, sentiment_result

    sentiment_result = analyze_sentiment(request.form.get('customer_statement',''))

    input_data = {}

    for col in model_columns:
        val = request.form.get(col)

        if col in label_encoders:
            try:
                val = label_encoders[col].transform([str(val)])[0]
            except:
                val = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]
        else:
            try:
                val = float(val)
            except:
                val = 0

        input_data[col] = val

    df_input = pd.DataFrame([input_data]).reindex(columns=model_columns, fill_value=0)
    scaled = scaler.transform(df_input)

    prob = model.predict_proba(scaled)[0][1]

    if sentiment_result["risk_flag"]:
        prob += 0.1

    threshold = 0.65

    if prob >= threshold:
        prediction_label = "Fraud"
        prediction_result = f"Fraud ❌ ({round(prob*100,2)}%)"
    else:
        prediction_label = "Safe"
        prediction_result = f"Safe ✅ ({round((1-prob)*100,2)}%)"

    return redirect('/performance')

@app.route('/performance')
@login_required
def performance():
    return render_template("performance.html",
        result=prediction_result,
        label=prediction_label,
        sentiment=sentiment_result,
        **metrics()
    )

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
