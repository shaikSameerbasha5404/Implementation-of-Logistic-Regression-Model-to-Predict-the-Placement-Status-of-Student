# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
```
import the standard libraries.
Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
Import LabelEncoder and encode the dataset.
Import LogisticRegression from sklearn and apply the model on the dataset.
Predict the values of array.
Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
Apply new unknown value
```

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Shaik Sameer Basha
RegisterNumber: 212222240093
```

```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") 
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]
```

## Output:
Original data(first five columns)

![4 1](https://github.com/shaikSameerbasha5404/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707756/deac9258-ddb8-4ffd-bd5a-8cc0ec0c35a4)



Data after dropping unwanted columns(first five):
![4 2](https://github.com/shaikSameerbasha5404/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707756/bcee46e8-7095-471e-b5df-1f908dd98600)


Checking the presence of null values:

![4 3](https://github.com/shaikSameerbasha5404/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707756/4e53f6c2-665d-4701-8061-27a4e032e08f)

Checking the presence of duplicated values

![4 4](https://github.com/shaikSameerbasha5404/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707756/7e66e141-d871-40be-bb5a-82a57fd64ae0)


Data after Encoding

![4 5](https://github.com/shaikSameerbasha5404/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707756/d9315256-87ef-4ded-bc98-67699dafe85b)


X Data
![4 6](https://github.com/shaikSameerbasha5404/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707756/e47a9ee6-f164-4e2a-ae74-bc9a43ac3460)


Y Data
![4 7](https://github.com/shaikSameerbasha5404/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707756/b89571b9-c5c5-43e9-b877-0c831b463b00)


Predicted Values

![4 8](https://github.com/shaikSameerbasha5404/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707756/8c060605-d831-4127-ac4b-58dfaed8ff72)

Accuracy Score


![4 9](https://github.com/shaikSameerbasha5404/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707756/d375fc2d-f6ed-45a1-acf5-a7c712c8f59d)

Confusion Matrix

![4 10](https://github.com/shaikSameerbasha5404/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707756/ce2abac4-32df-4372-8794-99e4584fb732)

Classification Report


![4 11](https://github.com/shaikSameerbasha5404/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707756/970598d4-6a5c-4d4e-959e-5bf8a57b8172)

Predicting output from Regression Model

![4 12](https://github.com/shaikSameerbasha5404/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707756/33f0553f-e253-4782-be30-da0a89e65a35)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
