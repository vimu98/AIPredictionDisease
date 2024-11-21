
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

heart_dataset = pd.read_csv('C:/Users/vimukthi/Desktop/prediction/dataset/heart.csv') 

X = heart_dataset.drop(columns = 'target', axis=1)
Y = heart_dataset['target']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, 
                                                    random_state=2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy score of the test data : ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy score of the test data : ', test_data_accuracy)


input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')
  
  
import pickle
  
filename = 'heart_disease_model.sav'
pickle.dump(model, open(filename, 'wb'))