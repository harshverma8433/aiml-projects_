from flask import Flask, render_template, request
import pandas as pd
import pickle

# Load the data and the pipeline
data = pd.read_csv("Cleaned_data.csv")
pipe = pickle.load(open("RidgeModel.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def index():
    locations = sorted(data["location"].unique())
    return render_template("index.html", locations=locations)

@app.route('/predict',methods=['POST'])
def predict():
    try:
        # Retrieve form data
        total_sqft = float(request.form.get('Squareft'))
        location = request.form.get('location')
        bhk = float(request.form.get('uiBHK'))
        bath = float(request.form.get('uiBathrooms'))

        print(location, total_sqft, bath, bhk)
        # Create a DataFrame with the correct columns
        prediction = pipe.predict(pd.DataFrame([[location, total_sqft, bath, bhk]],columns=["location", "total_sqft", "bath" , "bhk"]))[0]

        print("prediction", prediction)
        # return f"Rs. {prediction:.2f}"
        return ""
    except Exception as e:
        # Print the exception for debugging
        print("Error during prediction:")
        print(e)
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True, port=5001)
