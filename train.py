import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import string
import json

df = pd.read_csv("processed_data.csv")

df['message'].fillna('', inplace=True)


def text_process(title):
    nop = [char for char in title if char not in string.punctuation]
    nop = ''.join(nop)
    return [word for word in nop.split()]


pipelineMessage = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)),
])

X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=123)

y_pred = pipelineMessage.fit(X_train, y_train).predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred)

accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / (confusion_matrix[0]
                                                                [0] + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1])

# Now print to file test
with open("metrics.json", 'w') as outfile:
    json.dump({"accuracy": accuracy}, outfile)
