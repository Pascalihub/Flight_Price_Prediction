from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from src.flightprice.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            # Date_of_Journey
            date_dep = request.form["Dep_Time"],
            Journey_day=int(pd.to_datetime(request.form["Dep_Time"], format="%Y-%m-%dT%H:%M").day),
            Journey_month = int(pd.to_datetime(request.form["Dep_Time"], format ="%Y-%m-%dT%H:%M").month),
            # print("Journey Date : ",Journey_day, Journey_month)

            # Departure
            Dep_hour = int(pd.to_datetime(request.form["Dep_Time"], format ="%Y-%m-%dT%H:%M").hour),
            Dep_min = int(pd.to_datetime(request.form["Dep_Time"], format ="%Y-%m-%dT%H:%M").minute),
            # print("Departure : ",Dep_hour, Dep_min),

            # Arrival
            date_arr = request.form["Arrival_Time"],
            Arrival_hour = int(pd.to_datetime(request.form["Arrival_Time"], format ="%Y-%m-%dT%H:%M").hour),
            Arrival_min = int(pd.to_datetime(request.form["Arrival_Time"], format ="%Y-%m-%dT%H:%M").minute),
            # print("Arrival : ", Arrival_hour, Arrival_min),

            # Duration
            dur_hour = abs(Arrival_hour - Dep_hour),
            dur_min = abs(Arrival_min - Dep_min),
            # print("Duration : ", dur_hour, dur_min)

            Airline, 
    Source, 
    Destination, 
    Route, 
    Total_Stops, 
    Additional_Info, 
    Journey_day, 
    Journey_month, 
    Dep_hour, 
    Dep_min, 
    Arrival_hour, 
    Arrival_min, 
    Duration_hours, 
    Duration_mins

Multiple carriers  Delhi      Cochin
Jet Airways  Delhi      Cochin
SpiceJet
Multiple carriers
Air India
Air Asia
GoAir
Vistara  Kolkata    Banglore
IndiGo  Chennai     Kolkata
Jet Airways    Delhi      Cochin

            # Total Stops
            Total_stops = int(request.form["stops"]),
            # print(Total_stops)

            # Airline
            # AIR ASIA = 0 (not in column)
            airline=request.form['airline'],
            if(airline=='Jet Airways'):
                Jet_Airways = 1,
                IndiGo = 0,
                Air_India = 0,
                Multiple_carriers = 0,
                SpiceJet = 0,
                Vistara = 0,
                GoAir = 0,
                Multiple_carriers_Premium_economy = 0,
                Jet_Airways_Business = 0,
                Vistara_Premium_economy = 0,
                Trujet = 0 

            elif (airline=='IndiGo'):
                Jet_Airways = 0,
                IndiGo = 1,
                Air_India = 0,
                Multiple_carriers = 0,
                SpiceJet = 0,
                Vistara = 0,
                GoAir = 0,
                Multiple_carriers_Premium_economy = 0,
                Jet_Airways_Business = 0,
                Vistara_Premium_economy = 0,
                Trujet = 0 

            elif (airline=='Air India'):
                Jet_Airways = 0,
                IndiGo = 0,
                Air_India = 1,
                Multiple_carriers = 0,
                SpiceJet = 0,
                Vistara = 0,
                GoAir = 0,
                Multiple_carriers_Premium_economy = 0,
                Jet_Airways_Business = 0,
                Vistara_Premium_economy = 0,
                Trujet = 0 
            
            elif (airline=='Multiple carriers'):
                Jet_Airways = 0,
                IndiGo = 0,
                Air_India = 0,
                Multiple_carriers = 1,
                SpiceJet = 0,
                Vistara = 0,
                GoAir = 0,
                Multiple_carriers_Premium_economy = 0,
                Jet_Airways_Business = 0,
                Vistara_Premium_economy = 0,
                Trujet = 0 
            
            elif (airline=='SpiceJet'):
                Jet_Airways = 0,
                IndiGo = 0,
                Air_India = 0,
                Multiple_carriers = 0,
                SpiceJet = 1,
                Vistara = 0,
                GoAir = 0,
                Multiple_carriers_Premium_economy = 0,
                Jet_Airways_Business = 0,
                Vistara_Premium_economy = 0,
                Trujet = 0 
            
            elif (airline=='Vistara'):
                Jet_Airways = 0,
                IndiGo = 0,
                Air_India = 0,
                Multiple_carriers = 0,
                SpiceJet = 0,
                Vistara = 1,
                GoAir = 0,
                Multiple_carriers_Premium_economy = 0,
                Jet_Airways_Business = 0,
                Vistara_Premium_economy = 0,
                Trujet = 0

            elif (airline=='GoAir'):
                Jet_Airways = 0,
                IndiGo = 0,
                Air_India = 0,
                Multiple_carriers = 0,
                SpiceJet = 0,
                Vistara = 0,
                GoAir = 1,
                Multiple_carriers_Premium_economy = 0,
                Jet_Airways_Business = 0,
                Vistara_Premium_economy = 0,
                Trujet = 0

            elif (airline=='Multiple carriers Premium economy'):
                Jet_Airways = 0,
                IndiGo = 0,
                Air_India = 0,
                Multiple_carriers = 0,
                SpiceJet = 0,
                Vistara = 0,
                GoAir = 0,
                Multiple_carriers_Premium_economy = 1,
                Jet_Airways_Business = 0,
                Vistara_Premium_economy = 0,
                Trujet = 0

            elif (airline=='Jet Airways Business'):
                Jet_Airways = 0,
                IndiGo = 0,
                Air_India = 0,
                Multiple_carriers = 0,
                SpiceJet = 0,
                Vistara = 0,
                GoAir = 0,
                Multiple_carriers_Premium_economy = 0,
                Jet_Airways_Business = 1,
                Vistara_Premium_economy = 0,
                Trujet = 0

            elif (airline=='Vistara Premium economy'):
                Jet_Airways = 0,
                IndiGo = 0,
                Air_India = 0,
                Multiple_carriers = 0,
                SpiceJet = 0,
                Vistara = 0,
                GoAir = 0,
                Multiple_carriers_Premium_economy = 0,
                Jet_Airways_Business = 0,
                Vistara_Premium_economy = 1,
                Trujet = 0
            
            elif (airline=='Trujet'):
                Jet_Airways = 0,
                IndiGo = 0,
                Air_India = 0,
                Multiple_carriers = 0,
                SpiceJet = 0,
                Vistara = 0,
                GoAir = 0,
                Multiple_carriers_Premium_economy = 0,
                Jet_Airways_Business = 0,
                Vistara_Premium_economy = 0,
                Trujet = 1

            else:
                Jet_Airways = 0,
                IndiGo = 0,
                Air_India = 0,
                Multiple_carriers = 0,
                SpiceJet = 0,
                Vistara = 0,
                GoAir = 0,
                Multiple_carriers_Premium_economy = 0,
                Jet_Airways_Business = 0,
                Vistara_Premium_economy = 0,
                Trujet = 0

            # print(Jet_Airways,
            #     IndiGo,
            #     Air_India,
            #     Multiple_carriers,
            #     SpiceJet,
            #     Vistara,
            #     GoAir,
            #     Multiple_carriers_Premium_economy,
            #     Jet_Airways_Business,
            #     Vistara_Premium_economy,
            #     Trujet)

            # Source
            # Banglore = 0 (not in column)
            Source = request.form["Source"]
            if (Source == 'Delhi'):
                s_Delhi = 1,
                s_Kolkata = 0,
                s_Mumbai = 0,
                s_Chennai = 0,

            elif (Source == 'Kolkata'):
                s_Delhi = 0,
                s_Kolkata = 1,
                s_Mumbai = 0,
                s_Chennai = 0,

            elif (Source == 'Mumbai'):
                s_Delhi = 0,
                s_Kolkata = 0,
                s_Mumbai = 1,
                s_Chennai = 0

            elif (Source == 'Chennai'):
                s_Delhi = 0,
                s_Kolkata = 0,
                s_Mumbai = 0,
                s_Chennai = 1

            else:
                s_Delhi = 0,
                s_Kolkata = 0,
                s_Mumbai = 0,
                s_Chennai = 0

            # print(s_Delhi,
            #     s_Kolkata,
            #     s_Mumbai,
            #     s_Chennai)

            # Destination
            # Banglore = 0 (not in column)
            Source = request.form["Destination"]
            if (Source == 'Cochin'):
                d_Cochin = 1,
                d_Delhi = 0,
                d_New_Delhi = 0,
                d_Hyderabad = 0,
                d_Kolkata = 0,
        
            elif (Source == 'Delhi'):
                d_Cochin = 0,
                d_Delhi = 1,
                d_New_Delhi = 0,
                d_Hyderabad = 0,
                d_Kolkata = 0

            elif (Source == 'New_Delhi'):
                d_Cochin = 0,
                d_Delhi = 0,
                d_New_Delhi = 1,
                d_Hyderabad = 0,
                d_Kolkata = 0

            elif (Source == 'Hyderabad'):
                d_Cochin = 0,
                d_Delhi = 0,
                d_New_Delhi = 0,
                d_Hyderabad = 1,
                d_Kolkata = 0

            elif (Source == 'Kolkata'):
                d_Cochin = 0,
                d_Delhi = 0,
                d_New_Delhi = 0,
                d_Hyderabad = 0,
                d_Kolkata = 1

            else:
                d_Cochin = 0,
                d_Delhi = 0,
                d_New_Delhi = 0,
                d_Hyderabad = 0,
                d_Kolkata = 0
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")
    
