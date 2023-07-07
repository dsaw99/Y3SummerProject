from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS, cross_origin
import os
import openai
import dataProcessing as dataProcessing
import fridgeReport
import random
import pandas as pd

app = Flask(__name__)
app.secret_key = 'asdfghjrtyuio'
CORS(app)

nudges = [
    "Using a programmable thermostat could help you save up to £120 a year!",
    "Using energy-efficient lighting such as LEDs or CFLs could help you save up to £180 a year!",
    "Insulating your hot water tank could help you save up to £78 a year!",
    "Switching PC for a laptop could help you save up to £45 a year!",
    "Unplugging electronics vs. leaving them in standby could help you save up to £30 per device a year!",
    "Unplugging electronics using smart power strips could help you save up to £30 per device a year!",
    "Turning off lights / Using natural light could help you save up to £15 per room a year!",
    "Switching hot water tank for a tankless water heater could help you save up to £80 a year!",
    "Using thermal curtains, blinds, or shades for better insulation could help you save up to £60 a year!",
    "Swapping baths for showers could help you save up to £90 a year!",
    "Spending less time in the shower could help you save up to £90 a year!",
    "Installing double-glazed windows could help you save up to £50 a year!",
    "Insulating your attic and walls could help you save up to £240 a year!",
    "Insulating your hot water pipes could help you save up to £40 a year!"
]

openai.api_key = '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'

@app.route('/')
def home():
    if 'nudge_index' not in session:
        session['nudge_index'] = 0
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

@app.route('/api/setUser', methods=['POST'])
@cross_origin()
def dataAnalysis():
    user = request.json['user']
    data = dataProcessing.Dataset("output/User"+user+".csv")
    avg = data.getDailyAverageInterval(7)
    return jsonify(avg)

@app.route('/api/getScore', methods=['POST'])
@cross_origin()
def getScore():
    user = request.json['user']
    data = dataProcessing.Dataset("output/User"+user+".csv")
    score = data.getWeeklyRatio()
    return jsonify(score)

@app.route('/api/getChallenges', methods=['POST'])
@cross_origin()
def getChallenges():
    averages = dataProcessing.get_overall_average_consumption('output/', interval=30)
    print(averages)
    system_prompt = 'You are an expert analyst of energy consumption. You will be given household energy consumption features of a community of households and map it to \
        challenges in order to optimize/reduce energy consumption. \
        For example:  A community with high \"Oven\" consumption could get a creative but effective week-long Challenge to reduce that category of consumption, \
        eg. "You should avoid to use the oven this week, try new recipes that require either the microwave or stove top! You can expect a monthly savings of up to 10\% in the "Oven" category.". Be precise. '
    
    data_prompt = 'This community has the following daily consumption averages (in kWh): Total: ' + str(averages['Consumption']) + ', Fridge: ' +  str(averages['Fridge']) + ', Oven and other Heating Devices: ' \
       +  str(averages['Mystery Heat']) + ', Washing Machine and/or Dishwasher: ' +  str(averages['Mystery Motor']) + ', Tea Kettle: ' +  str(averages['Tea Kettle']) + \
      ', Freezer: ' +  str(averages['Freezer']) + ', Always On: ' +  str(averages['Always On']) + ', Microwave: ' +  str(averages['Microwave']) + ', Stove Top: ' + str(averages['Stove Top']) + ', Vaccum: ' + str(averages['Vacuum']) +'.'

    #data_prompt = 'This community has the following monthly consumption averages (in kWh): Always-On: 149.7, Washing Machine: 13.36, Dishwasher: 20.88, Laundry Dryer: 90.75 and Fridge: 19.89.'

    prompt = system_prompt + data_prompt + " \n What is 1 Effective and Feasible Challenge do you suggest for this community? The format of your answer should be: <Creative Title for Challenge>: <newline> \
        <Description of the Challenge>. <newline> <Predicted monthly energy savings as a percent reduction> in the category \"<Corresponding Category>\" . \
        \n Keep Your answer as truthful and scientifically accurate as possible and try not to create Challenges that require a big investment (>£10) for the Users. \
        Write as if you were directly talking with the household, as I did in the example."
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=400,
        temperature= 1
    )
    return jsonify(response['choices'][0]['text'].strip())

@app.route('/api/getConsumption', methods=['POST'])
@cross_origin()
def get_consumption():
    user = request.json['user']
    ds = dataProcessing.Dataset("output/User"+user+".csv")
    consumption_data = ds.get_consumption_last_7_days()
    return jsonify(consumption_data)


@app.route('/api/getConsumption2', methods=['POST'])
@cross_origin()
def get_consumption2():
    user = request.json['user']
    ds = dataProcessing.Dataset("output/User"+user+".csv")
    consumption_data = ds.get_consumption_last_7_days2()
    return jsonify(consumption_data)


@app.route('/get_hourly_data', methods=['GET'])
@cross_origin()
def get_hourly_data():
    user = request.args.get('user', None)
    date = request.args.get('date', None)
    device_type = request.args.get('type', None)

    if date is None or device_type is None:
        return jsonify({"error": "Missing date or type parameter"}), 400
    
    dataset = dataProcessing.Dataset("output/User"+user+".csv")
    data = dataset.get_hourly_bar_chart(date, device_type)
    return jsonify(data)

@app.route('/AlwaysOnRatio', methods=['POST'])
@cross_origin()
def isAlwaysOnRelevant():
    session['nudge_index'] = -1
    user = request.json['user']
    ds = dataProcessing.Dataset("output/User"+user+".csv")
    alwaysOnRelevancy = ds.getRelevancy('Always On')
    return jsonify(alwaysOnRelevancy)

@app.route('/NextNudge', methods=['POST'])
@cross_origin()
def NextNudge():
    session['nudge_index'] = (session['nudge_index'] + 1) % len(nudges)
    return jsonify(nudge=nudges[session['nudge_index']])

@app.route('/PreviousNudge', methods=['POST'])
@cross_origin()
def PreviousNudge():
    session['nudge_index'] = (session['nudge_index'] - 1) % len(nudges)
    return jsonify(nudge=nudges[session['nudge_index']])

@app.route('/generate_report', methods=['POST'])
@cross_origin()
def report():
    ReportClass = fridgeReport.Fridge()

    # Get data from form
    fridge_type = request.form.get('fridgeType')
    energy_rating = request.form.get('energyRating')
    household_size = int(request.form.get('householdSize'))
    years_used = int(request.form.get('yearsUsed'))

    # Construct the user dataframe
    users = pd.DataFrame({
        'User_ID': 1,
        'Fridge Type': [fridge_type],
        'Fridge Rating': [energy_rating],
        'Household_Size': [household_size],
        'Fridge Years Used': [years_used]
    })

    # Create empty dataframe for fridge_report and suggestion_list
    fridge_report = pd.DataFrame()
    fridge_suggestion_list = pd.DataFrame()

    # Call your function with this data
    fridge_list, group_size, total_saving = ReportClass.fridge_report(users, 1, fridge_report, fridge_suggestion_list)

    if fridge_list.empty:
        return "No recommendations for the User!"
    else:
        return render_template('report.html', fridge_list=fridge_list.to_dict('records'), group_size=group_size, total_saving=total_saving)
    
@app.route('/getSimulation', methods=['POST'])
@cross_origin()
def simulation():
    return render_template('simulation.html')

if __name__ == '__main__':
    app.run(port=5001)