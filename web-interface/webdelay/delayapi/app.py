from flask import Flask, request, render_template
from . import flightDelayPred

app = Flask(__name__, template_folder='templates')



@app.route('/')
def home():
	return render_template('index.html')
 
#delay_prediction(origin, destination,  carrier, dep_hour, arr_hour, month, day, weekday)

@app.route('/getdelay',methods=['POST','GET'])
def get_delay():
    if request.method=='POST':
        result=request.form
        origin = result['origin']
        destination = result['dest']
        carrier = result['unique_carrier']
        day = int(result['day'])
        month = int(result['month'])
        year = int(result['year'])
        dep_hour = int(result['dep_hour'])

        prediction, error = flightDelayPred.delay_prediction(origin, destination,  carrier, day, month, year, dep_hour)
        
        return render_template('result.html',prediction=prediction,error=error)