from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the CountVectorizer
with open('count_vec.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from the form (email title)
        input_data = request.form['input_data']
        
        # Transform the input text using the loaded CountVectorizer
        input_vector = vectorizer.transform([input_data])
        
        # Predict using the model
        prediction = model.predict(input_vector)

        #map prediction
        prediction_label ="Spam Email" if prediction[0]==1 else "Ham Email"
        
        # Render result
        return render_template('index.html', prediction=prediction_label , input_data=input_data)
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)


