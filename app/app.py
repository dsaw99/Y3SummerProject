from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS, cross_origin
import os
import openai
import dataProcessing as dataProcessing
import random

app = Flask(__name__)
app.secret_key = 'asdfghjrtyuio'
CORS(app)

@app.route('/')
def home():
    if 'username' in session and 'password' in session:
        if session['username'] == 'admin' and session['password'] == 'admin':
            return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data['username']
        password = data['password']

        if username == 'admin' and password == 'admin':
            session['username'] = username
            session['password'] = password
            return jsonify('success')

        else:
            return jsonify('Invalid username or password')
    else:
        return render_template('login.html', session=session)

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('password', None)
    return redirect(url_for('login'))


@app.route('/api/ask', methods=['POST'])
@cross_origin()
def ask():
    question = request.json['question']
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=question ,
        max_tokens=150
    )
    return jsonify(response['choices'][0]['text'].strip())

@app.route('/api/setUser', methods=['POST'])
@cross_origin()
def dataAnalysis():
    user = request.json['user']
    data = dataProcessing.Dataset("output/User"+user+".csv")
    avg = str(data.getDailyAverage())
    return jsonify("For User " + user + ", the average is " + avg + " kWh.")

@app.route('/api/getScore', methods=['POST'])
@cross_origin()
def getScore():
    user = request.json['user']
    data = dataProcessing.Dataset("output/User"+user+".csv")
    score = data.getScore()
    return jsonify({"score": score})

@app.route('/api/getChallenges', methods=['POST'])
@cross_origin()
def getChallenges():

    system_prompt = 'You are a system trained to analyze household energy consumption features of a community of households and map it to challenges in order to optimize/reduce energy consumption. \
        For example:  A community with high \"always-on\" consumption could get a creative but effective week-long Challenge to reduce that category of consumption.'

    data_prompt = 'This community has the following daily consumption averages: \n Oven: 3kWh,\n Microwave: 0.5 kWh, \n Dishwasher: 1kWh, \n Always-On: 5kWh, \n Total: 9.5kWh. \n \n '

    prompt = system_prompt + data_prompt + "What are the 5 Effective Challenges do you suggest for this community? The format of your answer should be: <Creative Title for Challenge>: \
          <Description of the Challenge>. <Potential daily energy savings> in the category \"<Corresponding Category>\""

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=400,
        temperature=0.2
    )
    return jsonify(response['choices'][0]['text'].strip())

if __name__ == '__main__':
    app.run(port=5001)



'''
@app.route('/api/getImage')
@cross_origin()
def get_image():
    image_path = os.path.join(os.path.dirname(__file__), 'boxplot.png')
    if not os.path.isfile(image_path):
        image_path = os.path.join(os.path.dirname(__file__), 'plot.png')  # serve a placeholder image
    return send_file(image_path, mimetype='image/png')

@app.route('/api/getAllBoxplot')
@cross_origin()
def get_allBoxplot():
    image_path = os.path.join(os.path.dirname(__file__), 'boxplotAll.png')
    return send_file(image_path, mimetype='image/png') '''