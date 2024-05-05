from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
import pickle


# create a flask application
app=Flask(__name__)

# Load the trained model
try:
    with open('heart_failure.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    print("Model file not found! Make sure 'heart_failure.pkl' exists in the current directory.")



@app.route('/')
def index():
    """
    Renders the index.html template when the user accesses the root URL.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles POST requests sent to the /predict route, makes predictions based on user input, 
    and renders the result.html template with the prediction result.
    """
    if request.method == 'POST':
        # Get user input from the form
        user_input = request.form
        features = [float(user_input['age']), int(user_input['anaemia']), int(user_input['creatinine_phosphokinase']), 
                    int(user_input['diabetes']), int(user_input['ejection_fraction']), int(user_input['high_blood_pressure']), 
                    float(user_input['platelets']), float(user_input['serum_creatinine']), int(user_input['serum_sodium']), 
                    int(user_input['sex']), int(user_input['smoking']), int(user_input['time'])]
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        # Display prediction result
        result = "Survived" if prediction == 0 else "Deceased"
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)