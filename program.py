import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train.fillna(method='bfill',inplace=True)
test.fillna(method='ffill',inplace=True)
test.fillna(method='bfill',inplace=True)

train['Age'] = train['Age'].astype(int)


test['Age'] = test['Age'].astype(int)


#encode as integer
mapping = {'male':0, 'female':1}
train = train.replace({'Sex':mapping})
test = test.replace({'Sex':mapping})

#encode as integer
mapping = {'C':0, 'Q':1, 'S':2}
train = train.replace({'Embarked':mapping})
test = test.replace({'Embarked':mapping})

#target variable
y_train = train['Survived']
test_id = test['PassengerId']

#drop columns
train.drop(['Survived','PassengerId','Ticket','Fare','Cabin','Name'], inplace=True, axis=1)
test.drop(['PassengerId','Ticket','Fare','Cabin','Name'],inplace=True,axis=1)

#train model
clf = RandomForestClassifier(n_estimators=500,max_features=3,min_samples_split=5,oob_score=True)
clf.fit(train, y_train)

#predict on test data
pred = clf.predict(test)

#write submission file and submit
columns = ['Survived']
sub = pd.DataFrame(data=pred, columns=columns)
sub['PassengerId'] = test_id
sub = sub[['PassengerId','Survived',]]
sub.to_csv("Survived.csv", index=False)
