import pandas as pd
from sklearn.linear_model import LogisticRegression

#%%
trainSet = pd.read_csv(r"C:\Users\Hieu\Desktop\titanic\train.csv")
testSet = pd.read_csv(r"C:\Users\Hieu\Desktop\titanic\test.csv")
passengerId = testSet['PassengerId']
y = trainSet.Survived

trainSet['Age'].fillna(trainSet['Age'].mean(), inplace=True)
testSet['Age'].fillna(testSet['Age'].mean(), inplace=True)
testSet['Fare'].fillna(testSet['Fare'].mean(), inplace=True)
print(trainSet.head())

features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
trainSet = trainSet.drop(['Name', 'Cabin', 'Survived', 'Ticket', 'PassengerId'], axis=1)
testSet = testSet.drop(['Name', 'Cabin', 'Ticket', 'PassengerId'], axis=1)

to_dummy_list = ['Sex', 'Embarked']

def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df

trainSet = dummy_df(trainSet,to_dummy_list)
testSet = dummy_df(testSet,to_dummy_list)
print(testSet.isnull().sum())

#%%


model = LogisticRegression(random_state=1, solver='lbfgs', max_iter=1000)
model.fit(trainSet, y)

predictions = model.predict(testSet)
print(predictions)
# output = pd.DataFrame({'PassengerId': passengerId,
#                         'Survived': predictions})
# output.to_csv(r'C:\Users\Hieu\submission.csv', index=False)






