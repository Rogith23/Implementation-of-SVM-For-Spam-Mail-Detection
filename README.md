# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Detect File Encoding: Use chardet to determine the dataset's encoding.
2.Load Data: Read the dataset with pandas.read_csv using the detected encoding.
3.Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().
4.Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.
5.Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.
6.Train SVM Model: Fit an SVC model on the training data.
7.Predict Labels: Predict test labels using the trained SVM model.
8.Evaluate Model: Calculate and display accuracy with metrics.accuracy_scor

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Rogith J
RegisterNumber: 212224040280
*/
import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect (rawdata.read(100000))
print(result) 
print("Name:Rogith")
print("Regno:212224040280")
import pandas as pd
data=pd.read_csv('spam.csv', encoding='Windows-1252')
print("Name:Rogith")
print("Regno:212224040280")
data.head()
print("Name:Rogith")
print("Regno:212224040280")
data.info()
print("Name:Rogith")
print("Regno:212224040280")
data.isnull().sum()
x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)
x_train
x_test
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
print("Name:Rogith")
print("Regno:212224040280")
x_train
print("Name:Rogith")
print("Regno:212224040280")
x_test
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train, y_train)
y_pred=svc.predict(x_test)
print("Name:Rogith")
print("Regno:212224040280")
y_pred
print("Name:Rogith")
print("Regno:212224040280")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
RESULT:

<img width="791" height="93" alt="Screenshot 2025-10-30 090930" src="https://github.com/user-attachments/assets/e72dece3-83ab-49b9-a022-ba03771908fa" />

HEAD:

<img width="793" height="287" alt="Screenshot 2025-10-30 090942" src="https://github.com/user-attachments/assets/b4a89438-fb28-47ff-ab09-3faa15a0c25b" />

INFO:

<img width="797" height="321" alt="Screenshot 2025-10-30 090951" src="https://github.com/user-attachments/assets/04b03b40-6bad-40d6-85d0-b97eece803cd" />

DATA.ISNULL,SUM()

<img width="867" height="209" alt="Screenshot 2025-10-30 091000" src="https://github.com/user-attachments/assets/50ef039f-b94f-43bc-a0e0-a52dfb3bc5dc" />

X_TRAIN:

<img width="865" height="117" alt="Screenshot 2025-10-30 091012" src="https://github.com/user-attachments/assets/5f8062c9-b160-496b-a325-6ee5e856fdb2" />

X_TEST:

<img width="902" height="154" alt="Screenshot 2025-10-30 091019" src="https://github.com/user-attachments/assets/1f971d03-d564-465c-864b-27db8a8c6bb0" />

Y_PRED:

<img width="835" height="101" alt="Screenshot 2025-10-30 091035" src="https://github.com/user-attachments/assets/79f2c1bb-0cc2-4b07-980d-c9d490d69390" />

ACCURACY:

<img width="1057" height="99" alt="Screenshot 2025-10-30 091043" src="https://github.com/user-attachments/assets/9fac9291-4581-497b-88d4-2a8e9944a02d" />







## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
