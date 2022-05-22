import numpy as np
import pickle
load_model=pickle.load(open('C:/Users/pabbu saikrishna/Desktop/deployment/trained_model','rb')) 
input_data=(5,106,72,19,175,25.0,0.587,51)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=load_model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")
 
