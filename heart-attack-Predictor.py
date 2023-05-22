import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import pickle

df=pd.read_csv('heart.csv')
df.info()

df.describe()

df['target'].value_counts()

sns.pairplot(df,hue='target')

X=df.drop('target',axis=1)
y=df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

sns.kdeplot(df['age'])

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

accuracy=classifier.score(X_train, y_train)
print(accuracy)

with open('heart_attack_model.pkl', 'wb') as file:
    pickle.dump(classifier, file)
