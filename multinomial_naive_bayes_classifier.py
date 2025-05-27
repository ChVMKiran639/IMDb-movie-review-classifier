import pandas as pd
data = pd.read_csv('datasets/imdb.csv')

x = data['review']
y = data['sentiment']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=60)

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(x_train)
x_train_df = vect.transform(x_train)
x_test_df = vect.transform(x_test)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train_df, y_train)
y_pred = model.predict(x_test_df)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy*100)

a=['Game Changer is good']
a = vect.transform(a)
print(model.predict(a))