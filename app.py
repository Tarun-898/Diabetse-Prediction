from flask import Flask, render_template, request
from predict import PredictionPipeline

app = Flask(__name__)
prediction_pipeline = PredictionPipeline()

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("index.html")

    try:
        features = {
            "age": int(request.form["age"]),
            "gender": request.form["gender"],
            "city": request.form["city"],
            "bmi": float(request.form["bmi"]),
            "family_history_diabetes": request.form["family_history_diabetes"],
            "physical_activity_level": request.form["physical_activity_level"],
            "diet_type": request.form["diet_type"],
            "smoking_status": request.form["smoking_status"],
            "alcohol_consumption": request.form["alcohol_consumption"],
            "hours_sleep_per_night": float(request.form["hours_sleep_per_night"]),
            "stress_level": int(request.form["stress_level"]),
            "fasting_blood_sugar": int(request.form["fasting_blood_sugar"]),
            "hba1c_level": float(request.form["hba1c_level"]),
            "blood_pressure_systolic": int(request.form["blood_pressure_systolic"]),
            "blood_pressure_diastolic": int(request.form["blood_pressure_diastolic"]),
            "waist_circumference_cm": float(request.form["waist_circumference_cm"]),
            "income_bracket": request.form["income_bracket"],
        }

        result = prediction_pipeline.predict(features)
        return render_template("index.html", result=result, form_data=features)

    except Exception as e:
        return render_template("index.html", error=str(e), form_data=request.form)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
