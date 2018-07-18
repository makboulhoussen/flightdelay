import pandas as pd
import datetime
import pickle
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import random


# Codes des compagnies aériennes et l'équivalent du code interne après numérisation de la feature.
carrier_dict = {'AA':0, 'AS':1, 'B6':2, 'DL':3, 'EV':4, 'F9':5,  'HA':6, 'NK':7, 'OO':8, 'UA':9,  'VX':10,'WN':11}

# Distance entre les destinations
tripDistances=pd.DataFrame()

# codes numérique et codes textuelles des aéroports 
airport_codes=pd.DataFrame()

# New Year day, Martin Luther King Jr. Day, Presidents' Day, Memorial Day
# Independence Day, Labor Day, Columbus Day, Veterans Day, 
# Thanksgiving, Christmas Day
holidays = [datetime.date(2018, 1, 1),datetime.date(2019, 1, 1), datetime.date(2020, 1, 1),
 datetime.date(2018, 1, 15),datetime.date(2019, 1, 21), datetime.date(2020, 1, 20),
 datetime.date(2018, 2, 19), datetime.date(2019, 2, 18), datetime.date(2020, 2, 17),
 datetime.date(2018, 5, 28), datetime.date(2019, 5, 27), datetime.date(2020, 5, 25),
 datetime.date(2018, 7, 4), datetime.date(2019, 7, 4), datetime.date(2020, 7, 4),
 datetime.date(2018, 9, 3), datetime.date(2019, 9, 2), datetime.date(2020, 9, 7),
 datetime.date(2018,10, 8), datetime.date(2019,10, 14), datetime.date(2020,10, 12),
 datetime.date(2018, 11, 11), datetime.date(2019, 11, 11), datetime.date(2020, 11, 11),
 datetime.date(2018, 11, 22), datetime.date(2019, 11, 28), datetime.date(2020, 11, 26), 
 datetime.date(2018, 12, 25), datetime.date(2019, 12, 25), datetime.date(2020, 12, 25)]

# Notre modèle de prédiction sauvegardé dans un fichier
predictionModel = SGDRegressor()

encoder = OneHotEncoder()
scaler = StandardScaler()

error_info = ''


def init(model_file='data/flights_delays_model.pkl', trip_distance_file='data/tripDistance.pkl', airport_code_file='data/airportCodesDF.pkl', encoder_file='data/categ_featuresEncoder.pkl', scaler_file='data/numfeaturesScaler.pkl') :
    global predictionModel, tripDistances, airport_codes,encoder, scaler
    predictionModel = joblib.load(model_file) 
    pkl_file = open(trip_distance_file, 'rb')
    tripDistances = pickle.load(pkl_file)    
    pkl_file = open(airport_code_file, 'rb')
    airport_codes =  pickle.load(pkl_file)
    encoder = joblib.load(encoder_file) 
    scaler = joblib.load(scaler_file)


# Retourne le numéro de semaine correspondant à la date
def getWeekNum(day, month,year) :
    global error_info
    try :
        fl_date =  datetime.date(year, month, day)
        return fl_date.isocalendar()[1]
    except Exception as err:
        error_info += 'Invalid date entered (' + str(day) + '/' + str(month) + '/' + str(year) + ') :' + str(err) + '. '
        raise(err)

# Retourne le jour de la semaine (1 = lundi, ...)	
def getWeekDay(day, month,year) :
    global error_info
    try :
        return datetime.date(year, month, day).weekday() + 1
    except Exception as err:
        error_info += 'Invalid date entered (' + str(day) + '/' + str(month) + '/' + str(year) + ') :' + str(err) + '. '
        raise(err)


# retourne le code numérique correspondant au code de la compagnies
def getCarrierCodeNum(unique_carrier_code):
    global error_info
    if  unique_carrier_code in carrier_dict :
        return carrier_dict[unique_carrier_code]
    else :
        error_info += 'Cannot find carrier code (' + unique_carrier_code + '). '
        raise ValueError('Bad carrier code')


# retourne la distance de vols entre 2 aéroports
def getTripDistance(origin_code, destination_code):
    global error_info
    try:
        distance = np.array(float(tripDistances[(tripDistances.ORIGIN == origin_code) & 
            (tripDistances.DEST == destination_code)].DISTANCE.drop_duplicates()))
        return distance
    except Exception as err:
        error_info += 'Route was not found in the data. Please try a different nearby city or a new route.'
        raise(err)
        


# Retourne le code numérique de l'aéoport d'origine (si true) ou destination si false.
def getAirportCodeNum(airport_code, origin=True):
    global error_info
    try :
        if origin :
            return int(airport_codes[airport_codes.AIRPORT_CODE == airport_code].ORIGIN_CODE)
        else :
            return int(airport_codes[airport_codes.AIRPORT_CODE == airport_code].DEST_CODE)
    except Exception as err:
        error_info += 'No airport found with code ' + str(airport_code) + '. '
        raise(err)



# Retourne le nombre de jour à proximité d'un jour férié
def getNumDaysToHoliday(day, month, year):
    if year not in [2018, 2019, 2020] :
        error_info += 'No data found for the year ' + str(year) + '. '
        raise ValueError('Bad year')
    c_date = datetime.date(year, month, day)
    return np.min(np.abs(np.array(c_date) - np.array(holidays))).days 


# Utilisation de notre modèle pour prédire le retard éventuel.
def delay_prediction(originCode, destCode,  carrier, day, month, year, dep_hour) :
    global error_info
    error_info=''
    try : 
        origin_code_num = getAirportCodeNum(originCode, True)
        dest_code_num = getAirportCodeNum(destCode, False)
        carrier_code_num = carrier_dict[carrier]
        weekday = getWeekDay(day, month, year)
        week_num = getWeekNum(day, month, year)
        hdays = getNumDaysToHoliday(day, month, year)
        distance = getTripDistance(originCode, destCode)

        numerical_values = np.c_[distance, hdays]
        # Scale the features
        numerical_values_scaled = scaler.transform(numerical_values)

        categorical_values = np.zeros(8)
        categorical_values[0] = int(month)
        categorical_values[1] = int(day)
        categorical_values[2] = int(weekday)
        categorical_values[3] = int(week_num)
        categorical_values[4] = int(dep_hour)
        categorical_values[5] = int(carrier_code_num)
        categorical_values[6] = int(origin_code_num)
        categorical_values[7] = int(dest_code_num)

        categorical_values_encoded = encoder.transform([categorical_values]).toarray()
        travel = np.c_[numerical_values_scaled, categorical_values_encoded]
        
        pred_delay = predictionModel.predict(travel)
        return int(pred_delay[0]),error_info
    except Exception as err:
        print(error_info)
        print ('Prediction error.', err)
        return None, error_info

def test() :
    tcarrier = ['AA', 'AS', 'DL', 'HA', 'UA']
    tday = [1,10, 6, 9, 23, 30, 26, 12, 6, 9]
    tmonth = [1,2, 3, 4, 5, 6, 7, 8, 9, 10,11,12]
    tcode = ['BOS', 'JFK', 'SEA', 'SAN', 'DCA']
    tdep_hour = [1, 2, 4, 7, 9, 12, 10, 15, 14, 17, 19, 20, 21, 22, 23]
    for i in range(1000) :
        origcode = random.choice(tcode)
        destcode = random.choice(tcode)
        carrier = random.choice(tcarrier)
        day = random.choice(tday)
        month = random.choice(tmonth)
        dep_hour = random.choice(tdep_hour) 
        d = delay_prediction(origcode, destcode, carrier, day, month, 2018, dep_hour)
        if d is not None :
            if d > 5 :
                print(origcode, destcode,carrier,day, month, dep_hour)
                print("delay", d)
                print("----------")

