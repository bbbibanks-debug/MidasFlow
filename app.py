import os
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Servidor Python criado direto pelo GitHub!"

if __name__ == '__main__':
    porta = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=porta)
