from flask import Flask, render_template, request
import pickle

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('spanmmm.pkl', 'rb'))

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    # Render the HTML form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the message from the form
    message = request.form.get('message', '')

    # Check if a message was provided
    if not message:
        return render_template('index.html', error="Please enter a message.")

    # Transform the message using the vectorizer
    transformed_message = tfidf.transform([message])

    # Predict using the loaded model
    prediction = model.predict(transformed_message)[0]
    prediction_label = "Spam" if prediction == 1 else "Not Spam"

    # Render the form with the result
    return render_template(
        'index.html',
        message=message,
        prediction=prediction_label
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
