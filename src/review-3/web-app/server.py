from flask import Flask,render_template,request,jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        print(request.json)
    return jsonify({'data':"12"})

if __name__ =='__main__':
    app.run(debug=True)
