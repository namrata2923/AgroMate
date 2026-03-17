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
import torch
from PIL import Image
from transformers import DeiTForImageClassification, DeiTImageProcessor, DeiTConfig
import requests

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

# Load disease DeiT model (PyTorch)
_disease_model_path = os.path.join(_MODELS_DIR, "best_deit_model1.pth")
_disease_names_path = os.path.join(_MODELS_DIR, "deit_class_names.pkl")

_NUM_CLASSES = 38
_device = torch.device("cpu")

if os.path.exists(_disease_model_path):
    disease_class_names = joblib.load(_disease_names_path)
    # Build model from config only (no pretrained weight download)
    _deit_config = DeiTConfig.from_pretrained("facebook/deit-tiny-patch16-224")
    _deit_config.num_labels = _NUM_CLASSES
    disease_model = DeiTForImageClassification(_deit_config)
    disease_model.load_state_dict(
        torch.load(_disease_model_path, map_location=_device)
    )
    disease_model.to(_device)
    disease_model.eval()
    _disease_processor = DeiTImageProcessor.from_pretrained("facebook/deit-tiny-patch16-224")
    print(f"[AgroMate] OK  DeiT disease model loaded -- {len(disease_class_names)} classes.")
else:
    disease_model = None
    disease_class_names = None
    _disease_processor = None
    print("[AgroMate] WARN Disease model not found -- place best_deit_model1.pth in models/ folder.")

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
    "Apple: Apple Scab", "Apple: Black Rot", "Apple: Cedar Apple Rust", "Apple: Healthy",
    "Blueberry: Healthy",
    "Cherry: Powdery Mildew", "Cherry: Healthy",
    "Corn: Cercospora Leaf Spot / Gray Leaf Spot", "Corn: Common Rust",
    "Corn: Northern Leaf Blight", "Corn: Healthy",
    "Grape: Black Rot", "Grape: Esca (Black Measles)",
    "Grape: Leaf Blight (Isariopsis Leaf Spot)", "Grape: Healthy",
    "Orange: Haunglongbing (Citrus Greening)",
    "Peach: Bacterial Spot", "Peach: Healthy",
    "Pepper Bell: Bacterial Spot", "Pepper Bell: Healthy",
    "Potato: Early Blight", "Potato: Late Blight", "Potato: Healthy",
    "Raspberry: Healthy",
    "Soybean: Healthy",
    "Squash: Powdery Mildew",
    "Strawberry: Leaf Scorch", "Strawberry: Healthy",
    "Tomato: Bacterial Spot", "Tomato: Early Blight", "Tomato: Late Blight",
    "Tomato: Leaf Mold", "Tomato: Septoria Leaf Spot",
    "Tomato: Spider Mites (Two-Spotted Spider Mite)",
    "Tomato: Target Spot", "Tomato: Yellow Leaf Curl Virus",
    "Tomato: Mosaic Virus", "Tomato: Healthy"
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
    """Run DeiT inference on uploaded leaf image bytes."""
    if disease_model is None or disease_class_names is None:
        raise RuntimeError("Disease model not loaded. Place best_deit_model1.pth in models/ folder.")
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = _disease_processor(images=img, return_tensors="pt").to(_device)
    with torch.no_grad():
        logits = disease_model(**inputs).logits          # shape: (1, 38)
    probs     = torch.softmax(logits, dim=1)[0].cpu().numpy()
    top_idx       = int(np.argmax(probs))
    disease_name  = _format_disease_name(disease_class_names[top_idx])
    confidence    = round(float(probs[top_idx]) * 100, 1)
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [
        (_format_disease_name(disease_class_names[i]), round(float(probs[i]) * 100, 1))
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


Weather_API_KEY = os.environ.get('Weather_API_KEY', '')

@app.route("/api/weather")
def get_weather_api():
    """JSON API for dynamic weather fetching from client-side geolocation."""
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    if not lat or not lon:
        return {"error": "lat and lon required"}, 400

    default_data = {
        "temperature": 28,
        "humidity": 65,
        "condition": "Partly Cloudy",
        "wind_speed": 14,
        "rainfall_chance": 30,
        "uv_index": 6
    }

    # Fetch real data if API key available
    if Weather_API_KEY:
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={Weather_API_KEY}&units=metric"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if "main" in data and "weather" in data and "wind" in data:
                return {
                    "temperature": round(data["main"]["temp"], 1),
                    "humidity": data["main"]["humidity"],
                    "condition": data["weather"][0]["description"].title(),
                    "wind_speed": round(data["wind"]["speed"], 1),
                    "rainfall_chance": 30,
                    "uv_index": 6
                }, 200
        except Exception as e:
            print(f"[AgroMate] Weather API error: {e}")
    
    return default_data, 200

@app.route("/weather-insights")
@login_required
def weather_insights():
    # Default demo data for initial page load
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

# ─── CACHE FOR MANDI PRICES DATA (Speeds up lookups) ──────────
# Stores all market records fetched from data.gov.in in memory
_MANDI_CACHE = []
_MANDI_CACHE_LOADED = False
_MANDI_CACHE_LOADING = False  # Flag to prevent multiple simultaneous loads

def _load_mandi_cache():
    """
    Fetch ALL mandi records from data.gov.in API once and cache in memory.
    This is called LAZILY on first request (not on startup) for faster startup.
    """
    global _MANDI_CACHE, _MANDI_CACHE_LOADED, _MANDI_CACHE_LOADING
    
    if _MANDI_CACHE_LOADED or _MANDI_CACHE_LOADING:
        return  # Already loaded or loading
    
    _MANDI_CACHE_LOADING = True
    print("[CACHE] Starting lazy load of mandi data...")
    
    try:
        if not DATA_GOV_API_KEY:
            print("[ERROR] DATA_GOV_API_KEY not configured - cache load failed")
            _MANDI_CACHE_LOADING = False
            return
        
        params = {
            "api-key": DATA_GOV_API_KEY,
            "format": "json",
            "limit": 10000,  # Get as many records as possible
            "offset": 0
        }
        
        resp = http_requests.get(
            f"https://api.data.gov.in/resource/{_MANDI_RESOURCE}",
            params=params,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        _MANDI_CACHE = data.get("records", [])
        _MANDI_CACHE_LOADED = True
        
        # Extract and log stats
        states = set(r.get("state", "").strip() for r in _MANDI_CACHE if r.get("state"))
        commodities = set(r.get("commodity", "").strip() for r in _MANDI_CACHE if r.get("commodity"))
        
        print(f"[CACHE] ✓ Lazy load complete: {len(_MANDI_CACHE)} records")
        print(f"[CACHE] ✓ States: {len(states)} | Commodities: {len(commodities)}")
        
    except http_requests.exceptions.Timeout:
        print("[ERROR] Cache load timeout - data.gov.in API took too long")
    except Exception as e:
        print(f"[ERROR] Failed to load mandi cache: {str(e)}")
    finally:
        _MANDI_CACHE_LOADING = False

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

    if request.method == "POST" or request.args.get("commodity"):
        commodity  = (request.form.get("commodity")   or request.args.get("commodity", "Tomato")).strip()
        state      = (request.form.get("state", "")   or request.args.get("state", "")).strip()

        if not state:
            error = "Please enter a state name."
        elif not _MANDI_CACHE:
            error = "Market data cache is not loaded yet. Please try again in a moment."
        else:
            # Search in cache (no API call!)
            print(f"[DEBUG] Cache Search: commodity='{commodity}', state='{state}'")
            print(f"[DEBUG] Cache contains {len(_MANDI_CACHE)} total records")
            
            # Filter cache by state and commodity (case-insensitive)
            records = [
                r for r in _MANDI_CACHE
                if r.get("state", "").strip().lower() == state.lower()
                and r.get("commodity", "").strip().lower() == commodity.lower()
            ]
            
            print(f"[DEBUG] Cache Results: {len(records)} records found")
            
            if not records:
                error = f"No market data found for '{commodity}' in {state}. Try checking the spelling or try a different commodity."
                
                # Suggest available states from cache
                available_states = sorted(list(set(
                    r.get("state", "").strip() for r in _MANDI_CACHE if r.get("state")
                )))
                print(f"[DEBUG] Available states in cache: {available_states}")
                if available_states:
                    error += f"\n\nAvailable states: {', '.join(available_states)}"

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
                        headers={"Content-Disposition": f"attachment;filename=market_prices_{commodity}_{state}.csv"})

    # Debug: Log what gets rendered
    print(f"[DEBUG] RENDER: commodity='{commodity}', state='{state}', records_count='{len(records)}'")
    
    return render_template(
        "market_prices.html",
        records=records,
        commodity=commodity,
        state=state,
        commodities=MARKET_COMMODITIES,
        error=error
    )


@app.route("/api/market-states", methods=["GET"])
@login_required
def get_market_states():
    """Fetch all available states from cache (instant - no API call)"""
    try:
        # Lazy load cache on first request
        if not _MANDI_CACHE_LOADED and not _MANDI_CACHE_LOADING:
            print("[DEBUG] get_market_states: Triggering lazy cache load...")
            _load_mandi_cache()
        
        if not _MANDI_CACHE:
            print("[DEBUG] get_market_states: Cache still loading or empty, returning loading status")
            return {"states": [], "loading": True, "message": "Loading market data... please wait"}, 202
        
        # Extract unique states from cache (instant lookup)
        states = sorted(list(set(r.get("state", "").strip() for r in _MANDI_CACHE if r.get("state"))))
        print(f"[DEBUG] get_market_states: Returning {len(states)} states from cache (instant)")
        
        return {"states": states, "count": len(_MANDI_CACHE), "loading": False}, 200
    
    except Exception as e:
        print(f"[ERROR] get_market_states: {str(e)}")
        return {"states": [], "error": str(e)}, 500


@app.route("/api/market-commodities", methods=["GET"])
@login_required
def get_market_commodities():
    """Fetch commodities available for a given state (from cache - instant)"""
    try:
        state = request.args.get("state", "").strip()
        
        if not state:
            return {"commodities": [], "error": "State parameter required"}, 400
        
        # Lazy load cache on first request
        if not _MANDI_CACHE_LOADED and not _MANDI_CACHE_LOADING:
            print("[DEBUG] get_market_commodities: Triggering lazy cache load...")
            _load_mandi_cache()
        
        if not _MANDI_CACHE:
            print("[DEBUG] get_market_commodities: Cache still loading or empty")
            return {"commodities": [], "loading": True, "message": "Loading market data... please wait"}, 202
        
        # Extract commodities for this state from cache (instant lookup)
        commodities = sorted(list(set(
            r.get("commodity", "").strip() 
            for r in _MANDI_CACHE 
            if r.get("state", "").strip().lower() == state.lower() and r.get("commodity")
        )))
        print(f"[DEBUG] get_market_commodities: {len(commodities)} commodities in {state} (instant from cache)")
        
        return {"commodities": commodities, "state": state, "count": len(commodities), "loading": False}, 200
    
    except Exception as e:
        print(f"[ERROR] get_market_commodities: {str(e)}")
        return {"commodities": [], "error": str(e)}, 500


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
#  Logbook Generation – Free APIs + Groq LLM
# ─────────────────────────────────────────────

import markdown as md_lib


def _geocode_location(location_name):
    """Geocode a location string → (lat, lon, display_name) using Open-Meteo Geocoding API (free, no key).

    Tries multiple search candidates to handle inputs like 'Pune, Maharashtra, India'
    (the API needs just the city name; state/country parts are stripped as fallbacks).
    """
    # Build a prioritised list of search terms:
    # 1. Full input  2. First comma-separated token (city)  3. First two tokens (city + state)
    candidates = [location_name.strip()]
    comma_parts = [p.strip() for p in location_name.split(",") if p.strip()]
    if len(comma_parts) >= 1 and comma_parts[0] not in candidates:
        candidates.append(comma_parts[0])
    if len(comma_parts) >= 2:
        two_part = f"{comma_parts[0]}, {comma_parts[1]}"
        if two_part not in candidates:
            candidates.append(two_part)

    for candidate in candidates:
        try:
            resp = http_requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": candidate, "count": 1, "language": "en", "format": "json"},
                timeout=10,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results:
                r = results[0]
                parts = [r.get("name", ""), r.get("admin1", ""), r.get("country", "")]
                display = ", ".join(p for p in parts if p)
                return float(r["latitude"]), float(r["longitude"]), display
        except Exception:
            continue

    return None, None, location_name


def _get_soil_data(lat, lon):
    """Fetch surface soil properties from ISRIC SoilGrids v2.0 (free, no API key)."""
    properties = ["phh2o", "nitrogen", "soc", "clay", "sand", "silt", "cec"]
    try:
        params = [("lat", lat), ("lon", lon), ("depth", "0-5cm"), ("value", "mean")]
        for p in properties:
            params.append(("property", p))
        resp = http_requests.get(
            "https://rest.isric.org/soilgrids/v2.0/properties/query",
            params=params,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        soil = {}
        for layer in data.get("properties", {}).get("layers", []):
            name       = layer.get("name", "")
            d_factor   = layer.get("unit_measure", {}).get("d_factor", 1) or 1
            target_units = layer.get("unit_measure", {}).get("target_units", "")
            depths     = layer.get("depths", [])
            if depths:
                val = depths[0].get("values", {}).get("mean")
                if val is not None:
                    soil[name] = {"value": round(val / d_factor, 2), "units": target_units}
        return soil, None
    except Exception as e:
        return {}, str(e)


def _get_weather_for_logbook(lat, lon):
    """Fetch current conditions + 7-day forecast from Open-Meteo (free, no API key)."""
    try:
        resp = http_requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":  lat,
                "longitude": lon,
                "current":   "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
                "daily":     "precipitation_sum,temperature_2m_max,temperature_2m_min,et0_fao_evapotranspiration",
                "timezone":  "auto",
                "forecast_days": 7,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return {}, str(e)


def _soil_summary(soil):
    if not soil:
        return "Soil data unavailable."
    labels = {
        "phh2o":    "Soil pH",
        "nitrogen": "Total Nitrogen",
        "soc":      "Organic Carbon",
        "clay":     "Clay Content",
        "sand":     "Sand Content",
        "silt":     "Silt Content",
        "cec":      "Cation Exchange Capacity (CEC)",
    }
    lines = []
    for key, label in labels.items():
        if key in soil:
            d = soil[key]
            lines.append(f"- {label}: {d['value']} {d['units']}")
    return "\n".join(lines) or "Soil data unavailable."


def _weather_summary(w):
    if not w:
        return "Weather data unavailable."
    cur   = w.get("current", {})
    daily = w.get("daily", {})
    lines = []
    if cur:
        lines.append(f"- Current Temperature: {cur.get('temperature_2m')}°C")
        lines.append(f"- Current Relative Humidity: {cur.get('relative_humidity_2m')}%")
        lines.append(f"- Current Precipitation: {cur.get('precipitation')} mm")
        lines.append(f"- Wind Speed: {cur.get('wind_speed_10m')} km/h")
    if daily:
        mt  = daily.get("temperature_2m_max", [])
        mn  = daily.get("temperature_2m_min", [])
        pr  = daily.get("precipitation_sum", [])
        et0 = daily.get("et0_fao_evapotranspiration", [])
        if mt:  lines.append(f"- 7-Day Avg Max Temperature: {round(sum(mt)/len(mt), 1)}°C")
        if mn:  lines.append(f"- 7-Day Avg Min Temperature: {round(sum(mn)/len(mn), 1)}°C")
        if pr:  lines.append(f"- 7-Day Total Rainfall: {round(sum(pr), 1)} mm")
        if et0: lines.append(f"- Daily Reference Evapotranspiration (ETo): {round(sum(et0)/len(et0), 2)} mm/day")
    return "\n".join(lines) or "Weather data unavailable."


_LOGBOOK_SYSTEM_PROMPT = """You are a friendly and knowledgeable farming advisor who helps Indian farmers. You write farm plans in simple, everyday language that any farmer can understand and follow — even someone who studied only up to class 8.

Rules:
- Write like a helpful friend, not a textbook. Use simple words.
- Avoid all technical jargon. Instead of "evapotranspiration", say "water the crop needs". Instead of "NPK", say "the nutrients that help the crop grow".
- Use Indian brand names (e.g. Urea, DAP, Bavistin, Confidor, Roger) that farmers recognise from the local kirana/agri shop.
- All doses must be practical: "mix 2 ml in 1 litre water" or "use 1 bag (50 kg) per acre" — not just kg/ha numbers.
- Always give quantities for BOTH 1 acre AND the farmer's total farm size.
- Use emojis (🌾 💧 ⚠️ ✅ 📅 💡) to make different sections easy to identify visually.
- Organise advice week by week so the farmer knows exactly what to do each week.
- Use Markdown formatting: ## for section headings, **bold** for important words, bullet points for lists.
- Amounts should be in Indian units: kg, bags (50 kg), litres, ml, and approximate Rupees (₹)."""


def _call_groq_for_logbook(prompt):
    """Call Groq LLM for logbook generation. Returns markdown-formatted text."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured. Add it to your .env file.")
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": _LOGBOOK_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.3,
        max_tokens=3000,
    )
    return response.choices[0].message.content.strip()


def _build_pesticide_prompt(crop, acres, location, soil_txt, weather_txt):
    return f"""You are writing a **Pesticide & Pest Control Weekly Schedule** for a farmer who grows {crop} on {acres} acres near {location}.

Write in simple, easy-to-understand language — as if you are a helpful knowledgeable friend explaining to a farmer with no technical background.
Do NOT use scientific jargon. Use everyday words.
Use emojis to make sections easy to identify.
All quantities must be given for BOTH 1 acre AND total for {acres} acres.
Mention popular Indian brand names (e.g. Dursban, Roger, Confidor, Bavistin, Mancozeb) alongside the product type.

**Soil information for this farm:**
{soil_txt}

**Current weather at this location:**
{weather_txt}

---

Generate the following sections:

## 🌿 What Pests & Diseases to Watch Out For
List in plain language the 4-5 most common problems for {crop} in this area. For each one, write:
- What it looks like on the plant (simple description a farmer can recognise)
- When it usually appears (which month or crop stage)
- How dangerous it is (mild / moderate / severe)

## 📅 Week-by-Week Spray Schedule
Write a simple weekly spray plan from sowing to harvest. For each entry use this format:

**Week [X] — [Crop Stage in plain words e.g. 'After sowing', 'When plant is knee-high']**
🎯 What to protect against: [pest/disease in plain words]
💊 What to spray: [Product type] — Indian brands: [Brand 1, Brand 2]
💧 How much to mix: [X ml / grams] in [Y litres of water] for 1 acre pump-tank
🌾 Total needed for your {acres} acres: [amount]
⏰ Best time to spray: [morning/evening]
⚠️ Stop spraying [X] days before harvest

Cover all important stages from Week 1 (after sowing) to the week before harvest.

## 🧴 How to Spray Correctly
Write 5-6 simple tips in bullet points. Example: "Always spray in the morning before 9 AM" or "Do not spray if it looks like rain".

## 🦺 Stay Safe While Spraying
Write 4-5 very simple safety rules. Example: "Cover your face with a cloth", "Wash hands after spraying", "Keep children away from the field".

## 💰 Rough Cost for the Season
List the main sprays needed and their approximate cost in Rupees (₹) for {acres} acres total.

> Write everything so that a farmer who has studied only up to class 8 can understand and follow it easily."""


def _build_fertilizer_prompt(crop, acres, location, soil_txt, weather_txt):
    return f"""You are writing a **Fertilizer & Nutrient Weekly Plan** for a farmer who grows {crop} on {acres} acres near {location}.

Write in simple, easy-to-understand language — as if you are a helpful knowledgeable friend explaining to a farmer with no technical background.
Do NOT use scientific jargon. Avoid terms like "NPK ratio", "CEC", "nitrification" — instead say "nitrogen helps the plant grow green and tall".
Use emojis to make sections easy to identify.
All quantities must be given for BOTH 1 acre AND total for {acres} acres.
Mention popular Indian brand names (e.g. Urea, DAP, MOP, 12:32:16, Polyfeed, Multifeed).

**Soil information for this farm (use this to personalise advice):**
{soil_txt}

**Current weather at this location:**
{weather_txt}

---

Generate the following sections:

## 🌱 What Your Soil Needs
Look at the soil data above and explain in 3-4 simple sentences what the soil is lacking or has enough of. Use language like: "Your soil has low nitrogen — this means your crop will grow slowly without extra help" or "Your soil is slightly acidic — this is okay for {crop}".

## 📅 Week-by-Week Fertilizer Plan
Write a simple weekly fertilizer plan from land preparation to just before harvest. For each entry use this format:

**Week [X] — [Stage in plain words e.g. 'Before sowing / When preparing the land', 'At sowing time', 'When plant has 4-5 leaves']**
🌾 What to apply: [Fertilizer name in common Indian name]
⚖️ How much for 1 acre: [amount in kg or bags]
🌾 Total for your {acres} acres: [amount]
✅ How to apply: [simple instruction e.g. "Spread evenly on the soil and mix it in" or "Dissolve in water and pour near plant roots"]
💡 Why: [one simple sentence explaining the benefit]

Cover: land preparation stage, sowing stage, early growth, mid-season, and flowering/fruiting stage.

## 🌿 Natural / Organic Options (Optional but Good)
Suggest 2-3 simple organic inputs a farmer can use (like vermicompost, cow dung, neem cake) with simple instructions.

## ⚠️ Important Do's and Don'ts
Write 5-6 simple rules. Example: "Do not apply urea when the field is flooded" or "Always water the field before applying dry fertilizer".

## 💰 Rough Cost for the Season
List the main fertilizers needed and their approximate cost in Rupees (₹) for {acres} acres total, with approximate bag/packet quantities.

> Write everything so that a farmer who has studied only up to class 8 can understand and follow it easily."""


def _build_irrigation_prompt(crop, acres, location, soil_txt, weather_txt):
    return f"""You are writing a **Water & Irrigation Weekly Plan** for a farmer who grows {crop} on {acres} acres near {location}.

Write in simple, easy-to-understand language — as if you are a helpful knowledgeable friend explaining to a farmer with no technical background.
Do NOT use scientific jargon. Avoid terms like "evapotranspiration", "soil matric potential", "hydraulic conductivity".
Use emojis to make sections easy to identify.
All quantities must be given for BOTH 1 acre AND total for {acres} acres.

**Soil information for this farm (affects how much water the soil holds):**
{soil_txt}

**Current weather at this location:**
{weather_txt}

---

Generate the following sections:

## 💧 How Much Water Does {crop} Need?
Write 2-3 simple sentences explaining the total water need for the season. Use easy comparisons like "This crop needs about X tanker-loads of water per acre during the whole season".

## 📅 Week-by-Week Irrigation Schedule
Write a simple weekly watering plan from sowing to harvest. For each entry use this format:

**Week [X] — [Crop Stage in plain words e.g. 'Just after sowing', 'When flowers start appearing']**
💧 Water now? [Yes — very important ⚠️ / Yes — normal / Can skip if it rained]
⏱️ How often: [Every X days]
🪣 How much water per acre: [in simple units like "fill the channel to 3 finger depth" or "approx X,000 litres"]
🌾 Total for your {acres} acres: [amount]
🌧️ If it rains: [simple rule e.g. "Skip this watering if you got good rain this week"]

Mark the stages where skipping water will damage the crop as **⚠️ CRITICAL — Do not skip**.

## 🚿 Which Irrigation Method is Best for You?
Explain in simple words the difference between flood irrigation, sprinkler, and drip irrigation. Tell the farmer which one suits {crop} and this soil type best, and roughly what each costs to set up per acre.

## 🌧️ What to Do When It Rains
Write 4-5 simple rules for adjusting watering when it rains. Example: "If it rains more than 25 mm, skip the next 2 waterings".

## 🔍 Simple Ways to Check If Your Soil Has Enough Water
Describe 2-3 simple field methods a farmer can do without any equipment. Example: "Take a fistful of soil from 6 inches deep — if it forms a ball and feels moist, you have enough water".

## ⚠️ Signs of Too Much or Too Little Water
List warning signs in simple language with what to do about it.

> Write everything so that a farmer who has studied only up to class 8 can understand and follow it easily."""


@app.route("/logbook-generation", methods=["GET", "POST"])
@login_required
def logbook_generation():
    result    = None
    error     = None
    form_data = {}

    if request.method == "POST":
        crop      = request.form.get("crop_name", "").strip()
        acres_str = request.form.get("acres", "").strip()
        location  = request.form.get("location", "").strip()
        lat_str   = request.form.get("lat", "").strip()
        lon_str   = request.form.get("lon", "").strip()
        form_data = {"crop_name": crop, "acres": acres_str, "location": location,
                     "lat": lat_str, "lon": lon_str}

        if not crop or not acres_str or (not lat_str and not location):
            error = "Please provide Crop Name, Acres, and either allow GPS access or enter a location."
        else:
            acres = None
            try:
                acres = float(acres_str)
                if acres <= 0:
                    error = "Acres must be greater than 0."
            except ValueError:
                error = "Acres must be a valid number (e.g. 5 or 2.5)."

            if not error and acres:
                # Step 1 — use GPS coordinates if provided, else geocode text location
                lat, lon, place_name = None, None, location
                if lat_str and lon_str:
                    try:
                        lat = float(lat_str)
                        lon = float(lon_str)
                        place_name = location if location else f"{lat:.4f}, {lon:.4f}"
                    except ValueError:
                        lat, lon = None, None

                if lat is None:
                    lat, lon, place_name = _geocode_location(location)
                    if lat is None:
                        error = (
                            f"Could not find '{location}'. "
                            "Please use a more specific name, e.g. 'Pune, Maharashtra, India'."
                        )

            if not error:
                # Step 2 — fetch soil data (non-fatal; proceed without it if unavailable)
                soil_data, _soil_err = _get_soil_data(lat, lon)
                soil_txt = _soil_summary(soil_data)

                # Step 3 — fetch weather data (non-fatal)
                weather_data, _wx_err = _get_weather_for_logbook(lat, lon)
                weather_txt = _weather_summary(weather_data)

                # Step 4 — generate all three logbooks via Groq LLM
                try:
                    pesticide_md  = _call_groq_for_logbook(
                        _build_pesticide_prompt(crop, acres, place_name, soil_txt, weather_txt))
                    fertilizer_md = _call_groq_for_logbook(
                        _build_fertilizer_prompt(crop, acres, place_name, soil_txt, weather_txt))
                    irrigation_md = _call_groq_for_logbook(
                        _build_irrigation_prompt(crop, acres, place_name, soil_txt, weather_txt))

                    result = {
                        "crop":       crop,
                        "acres":      acres,
                        "location":   place_name,
                        "soil":       soil_txt,
                        "weather":    weather_txt,
                        "pesticide":  md_lib.markdown(pesticide_md,  extensions=["extra"]),
                        "fertilizer": md_lib.markdown(fertilizer_md, extensions=["extra"]),
                        "irrigation": md_lib.markdown(irrigation_md, extensions=["extra"]),
                        "pesticide_md":  pesticide_md,
                        "fertilizer_md": fertilizer_md,
                        "irrigation_md": irrigation_md,
                    }
                    current_user.predictions += 1
                    db.session.commit()
                except Exception as e:
                    error = f"Logbook generation failed: {str(e)}"

    return render_template("logbook_generation.html",
                           result=result, error=error, form_data=form_data)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        # Cache loads lazily on first API request (not here on startup)
        print("[STARTUP] ✓ AgroMate server starting... (market data will load on first request)")
    app.run(debug=True)
