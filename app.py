from flask import Flask, request, jsonify

import joblib

app = Flask(__name__)

vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')

print("YAY ! Loaded")
@app.route('/' , methods=['GET'])
def health():
    return jsonify({'msg' : 'thank to god'})
@app.route('/analyze' , methods=['POST'])
def analyze_sentiment():
    try:

        data = request.get_json()

        if 'text' not in data :
            return jsonify({'error' : 'Missing "txt" key in request json'}), 400
        
        text_pre = data['text']
        text_pre_num = vectorizer.transform([text_pre])
        prd = model.predict(text_pre_num)

        sentiment = str(prd[0])

        return jsonify({'input_text' : text_pre, 'sentiment' : sentiment})
    
    except Exception as e:
        return jsonify({'error' : str(e)}) , 500
    


if __name__ == '__main__':
    
    app.run(port=5000, debug=True)