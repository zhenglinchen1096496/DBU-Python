# app.py
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "<h1>Welcome to the Homepage</h1><p>This is a truly simple Flask application.</p>"

@app.route('/about')
def about():
    return "<h1>About Us</h1><p>This page tells you more about us:)</p>"

@app.route('/new')
def home_new():
    return "<h1>Welcome to the Homepage</h1><p>This is a new application.</p>"

if __name__ == '__main__':
    app.run(debug=True)

