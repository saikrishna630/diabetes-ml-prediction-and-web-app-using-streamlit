import numpy as np
import pickle
import streamlit as st
#loading model
load_model=pickle.load(open('C:/Users/pabbu saikrishna/Desktop/deployment/trained_model','rb')) 
#creatinng function
def diabetes_prediction(input_data):
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    prediction=load_model.predict(input_data_reshaped)
    print(prediction)
    if(prediction[0]==0):
        return "The person is not diabetic"
    else:
        return "The person is diabetic"
def main():
    #giving a title
    st.title("DIABETES PREDICTION")
    #getting input data
    Pregnancies=st.text_input('Number of Pregnancies')
    Glucose=st.text_input('Glucose level')
    BloodPressure=st.text_input('Blood pressure value')
    SkinThickness=st.text_input('SkinThickness value')
    Insulin=st.text_input('Insulin level')
    BMI=st.text_input('BMI value')
    DiabetesPedigreeFunction=st.text_input('DiabetesPedigreeFunction value')
    Age=st.text_input('age of the person')
    
    #code for prediction
    diagnosis=" "
    #creating a button
    if st.button('Diabetes Test Result'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    st.success(diagnosis)
    
if __name__=='__main__':
    main()
    