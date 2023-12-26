import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv('mail_data.csv')
#print(df)    #print data set

data = df.where(pd.notnull(df))
#print(data.head(10))      #print top 5 lines or so  or top 10 rows so on

#print(data.info())         #info about dataset

#print(data.shape)     #matrix size

data.loc[data['Category'] == 'spam', 'Category',] =0

data.loc[data['Category'] == 'ham', 'Category',] =1  #not spam is ham

X=data['Message']
Y=data['Category']

#print(X)
#print(Y)

X_train, X_test , Y_train , Y_test = train_test_split(  X,Y,test_size=0.2 ,random_state = 3)  #ie 20 percent data tested and 80 persent used to train
#print(X.shape)
#print(X_train.shape)                        #80 percent
#print(X_test.shape)                       #20 percent


#transforming text into feature vector
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english',lowercase=1)
X_train_features= feature_extraction.fit_transform(X_train)
X_test_features= feature_extraction.transform(X_test)

Y_train= Y_train.astype('int')
Y_test= Y_test.astype('int')

#print(X_train)
#print(X_train_features)

#acccuracyy for traningggg dataaaaaaaaaa
model = LogisticRegression()
model.fit(X_train_features, Y_train)
prediction_on_traning_data=model.predict(X_train_features)
accuracy_on_training_data=accuracy_score(Y_train,prediction_on_traning_data)
print('Acc on traning data: ',accuracy_on_training_data)

#acccuracyy for testingggg dataaaaa
prediction_on_test_data=model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('acc on test data: ',accuracy_on_test_data)

input_ur_email=["URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18"]
input_data_features = feature_extraction.transform(input_ur_email)
prediction = model.predict(input_data_features)

print(prediction)
if(prediction[0]==1):
    print("ham mail")
else:
    print("spam")