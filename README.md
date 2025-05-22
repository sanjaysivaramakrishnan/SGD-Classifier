# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Libraries and Load iris Data set

2. Create a DataFrame from the Dataset

3. Add Target Labels to the DataFrame

4. Split Data into Features (X) and Target (y)

5. Split Data into Training and Testing Sets

6. Initialize the SGDClassifier Model

7. Train the Model on Training Data

8. Make Predictions on Test Data

9. Evaluate Accuracy of Predictions

10. Generate and Display Confusion Matrix

11. Generate and Display Classification Report

## Program:
```
Program to implement the prediction of iris species using SGD Classifier.
Developed by   :  Sanjay Sivaramakrishnan M
RegisterNumber :  212223240151
```

```
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import SGDClassifier
data_sets
data_sets = load_iris()
df = pd.DataFrame(data_sets.data,columns=data_sets.feature_names)
df['target'] = data_sets.target
df.head()
df.info()
X = df.drop(columns=['target'])
y = df['target']
#### Train Test split :
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
#### Model Implementation :
model = SGDClassifier(max_iter=1000,tol=3e-1)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy_score(y_pred,y_test)
confusion_matrix(y_pred,y_test)

```

## Output:

![image](https://github.com/user-attachments/assets/bfba204e-e69b-4a7c-aca3-e799402519a8)
![image](https://github.com/user-attachments/assets/d5bc2887-2089-4202-aa05-ca9cfcead7be)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
