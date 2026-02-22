from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import numpy as np
import joblib
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'agromate-secret-key-2026'
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
    print("[AgroMate] ✓ Crop model loaded successfully.")
else:
    print("[AgroMate] ⚠ Crop model not found — run train_crop_model.py first.")

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


def dummy_disease_predict(image_file):
    """Placeholder – swap with CNN model inference"""
    return DISEASE_LABELS[1], 87.4          # (disease_name, confidence %)


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
    confidence = None
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
                    result, confidence = dummy_disease_predict(file)
                    current_user.predictions += 1
                    db.session.commit()
        except Exception as e:
            error = str(e)
    return render_template("disease_detection.html",
                           result=result, confidence=confidence, error=error)


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


# ─────────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
