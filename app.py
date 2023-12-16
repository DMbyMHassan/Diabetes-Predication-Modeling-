from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the machine learning models
lg_model = joblib.load(
    "lg.pkl"
)  # Load the logistic regression model for diabetes prediction
mpg_model = joblib.load(
    "mpg_model.pkl"
)  # Load the linear regression model for MPG prediction


@app.route("/")
def home():
    # Render the home page
    return render_template("home.html")


@app.route("/mpg_prediction", methods=["GET", "POST"])
def mpg_prediction():
    if request.method == "POST":
        try:
            # Retrieve input data from the form
            cylinders = int(request.form["cylinders"])
            horsepower = float(request.form["horsepower"])
            weight = int(request.form["weight"])
            age = int(request.form["age"])
            origin_usa = bool(request.form.get("origin_usa"))
            origin_japan = bool(request.form.get("origin_japan"))

            # Make prediction using the MPG model
            input_data = np.array(
                [[cylinders, horsepower, weight, age, origin_usa, origin_japan]]
            )
            mpg_prediction = mpg_model.predict(input_data)

            # Render the prediction result page
            return render_template("mpg_prediction.html", prediction=mpg_prediction[0])
        except Exception as e:
            # Render the prediction result page with an error message in case of an exception
            return render_template("mpg_prediction.html", error_message=str(e))

    # Render the MPG prediction form page
    return render_template("mpg_prediction.html")


@app.route("/diabetes_prediction", methods=["GET", "POST"])
def diabetes_prediction():
    if request.method == "POST":
        try:
            # Retrieve input data from the form
            dpf = float(request.form["DiabetesPedigreeFunction"])
            bmi = float(request.form["BMI"])
            age = int(request.form["Age"])
            pregnancies = int(request.form["Pregnancies"])
            blood_pressure = int(request.form["BloodPressure"])
            glucose = int(request.form["Glucose"])
            insulin = int(request.form["Insulin"])

            # Make prediction using the Diabetes model
            # diabetes_prediction_proba = np.array([[dpf, bmi, age, pregnancies, blood_pressure, insulin, glucose]])
            # diabetes_prediction = 1 if diabetes_prediction_proba[0, 1] > 0.5 else 0

            input_data = np.array(
                [[dpf, bmi, age, pregnancies, blood_pressure, glucose, insulin]]
            )
            diabetes_prediction = lg_model.predict(input_data)

            # Render the prediction result page
            return render_template(
                "diabetes_prediction.html", prediction=diabetes_prediction
            )
        except Exception as e:
            print(f"Error: {str(e)}")
            # Render the prediction result page with an error message in case of an exception
            return render_template("diabetes_prediction.html", error_message=str(e))

    # Render the diabetes prediction form page
    return render_template("diabetes_prediction.html")


if __name__ == "__main__":
    # Run the Flask app in debug mode
    app.run(debug=True)
