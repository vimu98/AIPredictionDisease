
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_dataset = pd.read_csv('C:/Users/vimukthi/Desktop/prediction/dataset/lungCancer.csv') 
diabetes_dataset['LUNG_CANCER'].value_counts()
diabetes_dataset.groupby('LUNG_CANCER').mean()
X = diabetes_dataset.drop(columns = 'LUNG_CANCER', axis=1)
Y = diabetes_dataset['LUNG_CANCER']
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
Y = diabetes_dataset['LUNG_CANCER']


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, 
                                                    random_state=2)

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, Y_train)

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the train data : ', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)


input_data = (0,59,1,1,1,2,1,2,1,2,1,2,2,1,2)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not cancer')
else:
  print('The person is cancer')
  
  
import pickle
  
filename = 'cancer_model.sav'
pickle.dump(classifier, open(filename, 'wb'))