# Importing
from os import fsencode
from pyexpat import features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Loading the Datas
train_data = pd.read_csv("C:/Users/HP/Desktop/titanic/train.csv")
#print(train_data.head())

test_data = pd.read_csv("C:/Users/HP/Desktop/titanic/test.csv")
#print(test_data.head())


#women = train_data.loc[train_data.Sex == 'female']["Survived"]
#rate_women = sum(women)/len(women)
#print(" % of women who survived: ",rate_women)

#men = train_data.loc[train_data.Sex == 'male']["Survived"]
#rate_men = sum(men)/len(men)
#print(" % of men who survived: ",rate_men)

y = train_data['Survived']
features=['Pclass','Sex','SibSp','Parch']
X=pd.get_dummies(train_data[features])
X_test=pd.get_dummies(test_data[features])
model = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=1)
model.fit(X,y)
predictions = model.predict(X_test)
print(test_data.dtypes)
predictionId = test_data["PassengerId"].astype(str)
print(test_data.dtypes)
output = pd.DataFrame({'PassengerId': predictionId, 'Transported': predictions.astype(bool)})
output.to_csv('submissions.csv',index=False)
print("Saved")