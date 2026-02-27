from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
import joblib
import os
import io
import tensorflow as tf

load_dotenv()  # loads .env into environment variables

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'fallback-dev-key-change-me')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agromate.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# ─────────────────────────────────────────────
#  Load trained ML models
# ─────────────────────────────────────────────
_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def _load_model(filename):
    path = os.path.join(_MODELS_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None

crop_model         = _load_model("crop_model.pkl")
crop_label_encoder = _load_model("crop_label_encoder.pkl")

if crop_model:
    print("[AgroMate] OK  Crop model loaded successfully.")
else:
    print("[AgroMate] WARN Crop model not found -- run train_crop_model.py first.")

# Load disease CNN model
_disease_model_path = os.path.join(_MODELS_DIR, "disease_model.keras")
_disease_names_path = os.path.join(_MODELS_DIR, "disease_class_names.pkl")

if os.path.exists(_disease_model_path):
    disease_model = tf.keras.models.load_model(_disease_model_path)
    disease_class_names = joblib.load(_disease_names_path)
    print(f"[AgroMate] OK  Disease model loaded -- {len(disease_class_names)} classes.")
else:
    disease_model = None
    disease_class_names = None
    print("[AgroMate] WARN Disease model not found -- run train_disease_model.py first.")

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'warning'

# ─────────────────────────────────────────────
#  User Model
# ─────────────────────────────────────────────

class User(UserMixin, db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    full_name     = db.Column(db.String(120), nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    location      = db.Column(db.String(120), default='')
    farm_size     = db.Column(db.String(60), default='')
    crop_type     = db.Column(db.String(120), default='')
    joined_on     = db.Column(db.DateTime, default=datetime.utcnow)
    predictions   = db.Column(db.Integer, default=0)   # simple counter

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# ─────────────────────────────────────────────
#  Dummy ML prediction helpers
#  Replace these with your real trained models
# ─────────────────────────────────────────────

CROP_LABELS = [
    "Rice", "Maize", "Chickpea", "Kidney Beans", "Pigeon Peas",
    "Moth Beans", "Mung Bean", "Blackgram", "Lentil", "Pomegranate",
    "Banana", "Mango", "Grapes", "Watermelon", "Muskmelon",
    "Apple", "Orange", "Papaya", "Coconut", "Cotton",
    "Jute", "Coffee"
]

FERTILIZER_LABELS = [
    "Urea", "DAP", "14-35-14", "28-28", "17-17-17",
    "20-20", "10-26-26"
]

DISEASE_LABELS = [
    "Healthy", "Apple Scab", "Black Rot", "Cedar Apple Rust",
    "Bacterial Spot", "Early Blight", "Late Blight",
    "Leaf Mold", "Septoria Leaf Spot", "Spider Mites",
    "Target Spot", "Tomato Yellow Leaf Curl Virus", "Tomato Mosaic Virus"
]


def real_crop_predict(N, P, K, temperature, humidity, ph, rainfall):
    """Predict best crop using trained Random Forest model."""
    if crop_model is None or crop_label_encoder is None:
        raise RuntimeError("Crop model not loaded. Run train_crop_model.py first.")
    X = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    pred_enc      = crop_model.predict(X)[0]
    crop_name     = crop_label_encoder.inverse_transform([pred_enc])[0].title()
    probabilities = crop_model.predict_proba(X)[0]
    confidence    = round(float(max(probabilities)) * 100, 1)
    # Top-3 alternatives
    top3_idx  = np.argsort(probabilities)[::-1][:3]
    top3      = [
        (crop_label_encoder.inverse_transform([i])[0].title(),
         round(float(probabilities[i]) * 100, 1))
        for i in top3_idx
    ]
    return crop_name, confidence, top3


def dummy_fertilizer_predict(temperature, humidity, moisture, soil_type,
                              crop_type, N, P, K):
    """Placeholder – swap with model.predict()"""
    idx = int((N + P + K + temperature + humidity + moisture) % len(FERTILIZER_LABELS))
    return FERTILIZER_LABELS[idx]


def _format_disease_name(raw):
    """Convert folder names like Tomato__Early_blight → Tomato: Early Blight"""
    name = raw.replace("__", ": ").replace("_", " ")
    return name.title()


def real_disease_predict(image_bytes):
    """Run CNN inference on uploaded leaf image bytes."""
    if disease_model is None or disease_class_names is None:
        raise RuntimeError("Disease model not loaded. Run train_disease_model.py first.")
    img = tf.keras.utils.load_img(io.BytesIO(image_bytes), target_size=(224, 224))
    arr = tf.keras.utils.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = disease_model.predict(arr, verbose=0)[0]
    top_idx       = int(np.argmax(preds))
    disease_name  = _format_disease_name(disease_class_names[top_idx])
    confidence    = round(float(preds[top_idx]) * 100, 1)
    top3_idx = np.argsort(preds)[::-1][:3]
    top3 = [
        (_format_disease_name(disease_class_names[i]), round(float(preds[i]) * 100, 1))
        for i in top3_idx
    ]
    return disease_name, confidence, top3


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────

# ── Auth ──────────────────────────────────────

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == "POST":
        full_name  = request.form.get("full_name", "").strip()
        email      = request.form.get("email", "").strip().lower()
        password   = request.form.get("password", "")
        confirm    = request.form.get("confirm_password", "")
        location   = request.form.get("location", "").strip()
        farm_size  = request.form.get("farm_size", "").strip()
        crop_pref  = request.form.get("crop_type", "").strip()

        if not full_name or not email or not password:
            flash("All required fields must be filled in.", "danger")
        elif password != confirm:
            flash("Passwords do not match.", "danger")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
        elif User.query.filter_by(email=email).first():
            flash("An account with that email already exists. Please log in.", "warning")
            return redirect(url_for('login'))
        else:
            user = User(
                full_name=full_name, email=email,
                location=location, farm_size=farm_size, crop_type=crop_pref
            )
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            login_user(user)
            flash(f"Welcome to AgroMate, {user.full_name}! Your account has been created.", "success")
            return redirect(url_for('profile'))
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        remember = True if request.form.get("remember") else False
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user, remember=remember)
            flash(f"Welcome back, {user.full_name}!", "success")
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash("Invalid email or password. Please try again.", "danger")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for('index'))


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        current_user.full_name = request.form.get("full_name", current_user.full_name).strip()
        current_user.location  = request.form.get("location", "").strip()
        current_user.farm_size = request.form.get("farm_size", "").strip()
        current_user.crop_type = request.form.get("crop_type", "").strip()
        # password change (optional)
        new_pass    = request.form.get("new_password", "")
        confirm_new = request.form.get("confirm_new_password", "")
        if new_pass:
            if len(new_pass) < 6:
                flash("New password must be at least 6 characters.", "danger")
                return redirect(url_for('profile'))
            if new_pass != confirm_new:
                flash("New passwords do not match.", "danger")
                return redirect(url_for('profile'))
            current_user.set_password(new_pass)
        db.session.commit()
        flash("Profile updated successfully!", "success")
        return redirect(url_for('profile'))
    return render_template("profile.html")


# ── Public pages ──────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


# ── Protected ML Services ─────────────────────

@app.route("/crop-prediction", methods=["GET", "POST"])
@login_required
def crop_prediction():
    result = None
    error = None
    if request.method == "POST":
        try:
            N           = float(request.form["nitrogen"])
            P           = float(request.form["phosphorus"])
            K           = float(request.form["potassium"])
            temperature = float(request.form["temperature"])
            humidity    = float(request.form["humidity"])
            ph          = float(request.form["ph"])
            rainfall    = float(request.form["rainfall"])
            crop_name, confidence, top3 = real_crop_predict(
                N, P, K, temperature, humidity, ph, rainfall
            )
            result = {"crop": crop_name, "confidence": confidence, "top3": top3}
            current_user.predictions += 1
            db.session.commit()
        except Exception as e:
            error = str(e)
    return render_template("crop_prediction.html", result=result, error=error)


@app.route("/fertilizer-recommendation", methods=["GET", "POST"])
@login_required
def fertilizer_recommendation():
    result = None
    error = None
    soil_types = ["Sandy", "Loamy", "Black", "Red", "Clayey"]
    crop_types = ["Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy",
                  "Barley", "Wheat", "Millets", "Oil seeds", "Pulses", "Ground Nuts"]
    if request.method == "POST":
        try:
            temperature = float(request.form["temperature"])
            humidity    = float(request.form["humidity"])
            moisture    = float(request.form["moisture"])
            soil_type   = request.form["soil_type"]
            crop_type   = request.form["crop_type"]
            N           = float(request.form["nitrogen"])
            P           = float(request.form["phosphorus"])
            K           = float(request.form["potassium"])
            result = dummy_fertilizer_predict(temperature, humidity, moisture,
                                              soil_type, crop_type, N, P, K)
            current_user.predictions += 1
            db.session.commit()
        except Exception as e:
            error = str(e)
    return render_template("fertilizer_recommendation.html",
                           result=result, error=error,
                           soil_types=soil_types, crop_types=crop_types)


@app.route("/disease-detection", methods=["GET", "POST"])
@login_required
def disease_detection():
    result = None
    error = None
    if request.method == "POST":
        try:
            if "plant_image" not in request.files:
                error = "No file uploaded."
            else:
                file = request.files["plant_image"]
                if file.filename == "":
                    error = "No file selected."
                else:
                    image_bytes = file.read()
                    disease_name, confidence, top3 = real_disease_predict(image_bytes)
                    result = {"disease": disease_name, "confidence": confidence, "top3": top3}
                    current_user.predictions += 1
                    db.session.commit()
        except Exception as e:
            error = str(e)
    return render_template("disease_detection.html",
                           result=result, error=error)


@app.route("/weather-insights")
@login_required
def weather_insights():
    # Static demo data – integrate a real weather API here
    weather_data = {
        "city": "Pune, Maharashtra",
        "temperature": 28,
        "humidity": 65,
        "condition": "Partly Cloudy",
        "wind_speed": 14,
        "rainfall_chance": 30,
        "uv_index": 6,
        "forecast": [
            {"day": "Mon", "icon": "cloud-sun", "high": 30, "low": 22},
            {"day": "Tue", "icon": "cloud-rain", "high": 27, "low": 20},
            {"day": "Wed", "icon": "sun",        "high": 32, "low": 23},
            {"day": "Thu", "icon": "cloud",      "high": 29, "low": 21},
            {"day": "Fri", "icon": "cloud-sun",  "high": 31, "low": 22},
        ]
    }
    return render_template("weather_insights.html", weather=weather_data)


# ── Crop Market Prices ────────────────────────

import requests as http_requests

DATA_GOV_API_KEY = os.environ.get('DATA_GOV_API_KEY', '')

# data.gov.in resource ID for daily mandi prices
_MANDI_RESOURCE = "9ef84268-d588-465a-a308-a864a43d0070"

# Commodity choices shown in the UI
MARKET_COMMODITIES = [
    "Tomato", "Potato", "Onion", "Rice", "Wheat", "Maize",
    "Soyabean", "Cotton", "Sugarcane", "Groundnut", "Mustard",
    "Brinjal", "Cabbage", "Cauliflower", "Garlic", "Ginger",
    "Green Chilli", "Banana", "Mango", "Grapes"
]


MARKET_SORT_FIELDS = [
    ("market",        "Market"),
    ("state",         "State"),
    ("district",      "District"),
    ("commodity",     "Commodity"),
    ("variety",       "Variety"),
    ("arrival_date",  "Arrival Date"),
    ("min_price",     "Min Price"),
    ("max_price",     "Max Price"),
    ("modal_price",   "Modal Price"),
]

@app.route("/market-prices", methods=["GET", "POST"])
@login_required
def market_prices():
    records    = []
    error      = None
    commodity  = "Tomato"
    state      = ""
    district   = ""
    date_from  = ""
    date_to    = ""
    sort_by    = "market"
    sort_order = "asc"

    if request.method == "POST" or request.args.get("commodity"):
        commodity  = (request.form.get("commodity")   or request.args.get("commodity", "Tomato")).strip()
        state      = (request.form.get("state", "")   or request.args.get("state", "")).strip()
        district   = (request.form.get("district", "") or request.args.get("district", "")).strip()
        date_from  = (request.form.get("date_from", "") or request.args.get("date_from", "")).strip()
        date_to    = (request.form.get("date_to", "")   or request.args.get("date_to", "")).strip()
        sort_by    = (request.form.get("sort_by", "market") or "market").strip()
        sort_order = (request.form.get("sort_order", "asc") or "asc").strip()

        if not DATA_GOV_API_KEY:
            error = "DATA_GOV_API_KEY is not configured. Add it to your .env file."
        else:
            try:
                # HTML date inputs give yyyy-mm-dd; API expects dd/mm/yyyy
                def fmt_date(d):
                    try:
                        from datetime import datetime
                        return datetime.strptime(d, "%Y-%m-%d").strftime("%d/%m/%Y")
                    except Exception:
                        return d

                params = {
                    "api-key":            DATA_GOV_API_KEY,
                    "format":             "json",
                    "limit":              100,
                    "filters[commodity]": commodity,
                }
                if state:
                    params["filters[state]"] = state
                if district:
                    params["filters[district]"] = district
                if date_from and date_to:
                    params["filters[arrival_date][from]"] = fmt_date(date_from)
                    params["filters[arrival_date][to]"]   = fmt_date(date_to)
                elif date_from:
                    params["filters[arrival_date][from]"] = fmt_date(date_from)
                elif date_to:
                    params["filters[arrival_date][to]"]   = fmt_date(date_to)
                if sort_by:
                    params["sort[]"]     = sort_by
                    params["sort_order"] = sort_order

                resp = http_requests.get(
                    f"https://api.data.gov.in/resource/{_MANDI_RESOURCE}",
                    params=params,
                    timeout=10
                )
                resp.raise_for_status()
                data = resp.json()
                records = data.get("records", [])
                if not records:
                    location = f" in {district}" if district else (f" in {state}" if state else "")
                    date_hint = " Today's data may not be submitted yet — try yesterday's date." if date_from else ""
                    error = f"No market data found for '{commodity}'{location}.{date_hint} Try adjusting your filters."
            except http_requests.exceptions.Timeout:
                error = "Request timed out. The data.gov.in API is slow right now — please try again."
            except Exception as e:
                error = f"Could not fetch market data: {str(e)}"

    # CSV export
    if records and request.args.get("export") == "csv":
        import csv, io
        si = io.StringIO()
        writer = csv.DictWriter(si, fieldnames=["state","district","market","commodity","variety","arrival_date","min_price","max_price","modal_price"])
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
        output = si.getvalue()
        from flask import Response
        return Response(output, mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment;filename=market_prices_{commodity}.csv"})

    return render_template(
        "market_prices.html",
        records=records,
        commodity=commodity,
        state=state,
        district=district,
        date_from=date_from,
        date_to=date_to,
        sort_by=sort_by,
        sort_order=sort_order,
        commodities=MARKET_COMMODITIES,
        sort_fields=MARKET_SORT_FIELDS,
        error=error
    )


# ── Chatbot (Groq – llama-3.3-70b-versatile) ──

from flask import jsonify
from groq import Groq

# ── !! Paste your Groq API key below !! ─────────────────────────
#   Get it free at: https://console.groq.com → API Keys
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
# ────────────────────────────────────────────────────────────────

GROQ_MODEL = "llama-3.3-70b-versatile"

# System prompt — gives the LLM its identity and scope
_SYSTEM_PROMPT = """You are AgroBot 🌿, a friendly and knowledgeable AI agriculture assistant for AgroMate — a smart farming platform based in India.

Your role:
- Answer questions about crops, soil health, fertilizers, plant diseases, irrigation, pest control, weather-smart farming, organic farming, and government agricultural schemes.
- When relevant, mention AgroMate's built-in tools: Crop Prediction, Fertilizer Recommendation, Disease Detection (upload a leaf photo), and Weather Insights.
- Give practical, actionable advice suited for Indian farmers.
- Be concise but helpful. Use bullet points and emojis where appropriate to keep responses readable.
- If a question is completely unrelated to agriculture or farming, politely redirect the user back to agriculture topics.
- Always respond in the same language the user writes in (English or Hindi).
- Never make up scientific facts. If unsure, say so and suggest consulting a local agronomist or Krishi Vigyan Kendra (KVK).
"""

# Per-session conversation history (in-memory, resets on server restart)
# Key: session_id (we use a simple per-request approach — stateless is fine for MVP)
_chat_histories: dict = {}


def _get_groq_reply(message: str, history: list) -> str:
    """Send message + history to Groq and return the assistant reply."""
    client = Groq(api_key=GROQ_API_KEY)

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


@app.route("/chatbot", methods=["POST"])
def chatbot():
    data       = request.get_json(silent=True) or {}
    message    = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    if not message:
        return jsonify({"reply": "Please type a message! 🌱"})

    if not GROQ_API_KEY:
        return jsonify({"reply": "⚠️ AgroBot is not configured yet. Please add your Groq API key in app.py to enable AI chat."})

    # Maintain per-session history (last 10 exchanges = 20 messages)
    if session_id not in _chat_histories:
        _chat_histories[session_id] = []
    history = _chat_histories[session_id]

    try:
        reply = _get_groq_reply(message, history)
        # Append to history
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant", "content": reply})
        # Keep only last 20 messages to stay within token limits
        _chat_histories[session_id] = history[-20:]
        return jsonify({"reply": reply})
    except Exception as e:
        error_msg = str(e)
        if "invalid_api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            return jsonify({"reply": "⚠️ Invalid Groq API key. Please check the key in app.py."})
        if "rate_limit" in error_msg.lower():
            return jsonify({"reply": "⏳ Too many requests. Please wait a moment and try again!"})
        return jsonify({"reply": f"⚠️ Sorry, something went wrong. Please try again. ({error_msg[:80]})"})



# ─────────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
