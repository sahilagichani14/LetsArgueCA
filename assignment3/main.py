import numpy as np
import pandas as pd
import json
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.svm import SVC
import spacy
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences

nlp = spacy.load('en_core_web_lg')
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

def get_nooftokens(x):
    count = 0
    for dic in x:
        for key, value in dic.items():
            if key=='text':
                re.sub(r'[^\w ]+', " ", dic[key]) #remove special words or punctuations
                ' '.join(dic[key].split()) #remove multiple spaces
                str(dic[key]).lower() #convert into lower case
                ' '.join([t for t in dic[key].split() if t not in stopwords])
                doc = nlp(dic[key])
                count = count+ len(nlp(dic[key]))
    return count

def get_vec(x):
    doc = nlp(str(x))
    vec = doc.vector
    return vec

glove_vectors = dict()
file = open('./glove/glove.6B.100d.txt', encoding='ISO-8859-1')

for line in file:
    values = line.split()
    word  = values[0]
    vectors = np.asarray(values[1:])
    glove_vectors[word] = vectors
        
file.close()

def get_glove_vec(x):
    arr = np.zeros(100)
    text = str(x).split()
    
    for t in text:
        try:
            vec = glove_vectors.get(t).astype(float)
            arr = arr + vec
        except:
            pass
        
    arr = arr.reshape(1, -1)[0]
    return arr/len(text)

lstofpolaritydiffvec = []

def get_sentimentdiff(x):
    scr_arr = []
    for dic in x:
        for key, value in dic.items():
            if key=='text':
                re.sub(r'[^\w ]+', " ", dic[key]) #remove special words or punctuations
                ' '.join(dic[key].split()) #remove multiple spaces
                str(dic[key]).lower() #convert into lower case
                ' '.join([t for t in dic[key].split() if t not in stopwords]) #remove stop words
                polarity_dic = sid.polarity_scores(dic[key])
                scr_arr.append(polarity_dic['compound'])
    
    polaritydiff_vec = []
    for i in range(1,len(scr_arr)):
        polaritydiff_vec.append(abs(scr_arr[i]-scr_arr[i-1]))
    
    lstofpolaritydiffvec.append(polaritydiff_vec)

    return polaritydiff_vec

def main():
    
    df=pd.read_json('./data/essay-corpus.json', encoding='ISO-8859-1')
    
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^\w ]+', " ", x)) #remove special words or punctuations
    df['text'] = df['text'].apply(lambda x: ' '.join(x.split())) #remove multiple spaces
    df['text'] = df['text'].apply(lambda x: str(x).lower()) #convert into lower case

    df['text_sentiment_scores'] = df['text'].apply(lambda text: sid.polarity_scores(text)) #calculating sentiment scr of text
    df['text_sentiment_scr']  = df['text_sentiment_scores'].apply(lambda score_dict: score_dict['compound']) #getting the compound of pos,neg,neu

    df['vec'] = df['text'].apply(lambda x: np.array(get_vec(x))) #converting text to vec using spacy 
    df['text'] = df['text'].apply(lambda x: ' '.join([t for t in x.split() if t not in stopwords])) #remove stop words
    df['no_of_major_claims'] = df['major_claim'].apply(lambda x: len(x))
    df['no_of_claims'] = df['claims'].apply(lambda x: len(x))
    df['no_of_premises'] = df['premises'].apply(lambda x: len(x))
                   
    df['nooftokens_majorclaim'] = df['major_claim'].apply(lambda x: get_nooftokens(x))
    df['nooftokens_claims'] = df['claims'].apply(lambda x: get_nooftokens(x))
    df['nooftokens_premises'] = df['premises'].apply(lambda x: get_nooftokens(x))

    df['confirmation_bias'] = df['confirmation_bias'].astype(int) #converting true false to 0/1
    
    df['glove_vec'] = df['text'].apply(lambda x: get_glove_vec(x)) #also using glove vec of 100 dimentions

    df['claims_polaritydiff_vec'] = df['claims'].apply(lambda x: get_sentimentdiff(x))
    df['premises_polaritydiff_vec'] = df['premises'].apply(lambda x: get_sentimentdiff(x))

    claims_lstofpolaritydiffvec = df['claims_polaritydiff_vec'].tolist()
    max_len = max([len(x) for x in claims_lstofpolaritydiffvec])
    polarity_pad_vec = pad_sequences(claims_lstofpolaritydiffvec, maxlen=max_len, padding = 'post', dtype='float') #adding 0's to make all of same max length
    
    temp_df = pd.DataFrame(polarity_pad_vec).agg(lambda x: x.values, axis=1).T
    new_df = pd.concat([df, temp_df], axis=1)
    new_df.rename(columns = {0:'claims_padded_polaritydiff_vec'}, inplace = True)

    splits = pd.read_csv("./data/train-test-split.csv", sep=";", encoding='ISO-8859-1') #getting train & test data
    train_ids = sorted([int(fn[-3:]) for fn in splits[splits.SET == "TRAIN"].ID.values])
    test_ids = sorted([int(fn[-3:]) for fn in splits[splits.SET == "TEST"].ID.values])

    train_df = new_df[new_df['id'].isin(train_ids)] #train data
    test_df = new_df[new_df['id'].isin(test_ids)] #test data

    vectorizer = TfidfVectorizer(max_features = 1000, min_df = 0.02, norm='l1', ngram_range=(1,2), analyzer='word')
    uvw = vectorizer.fit_transform(train_df['text']).toarray()
    uvw_test = vectorizer.transform(test_df['text']).toarray()

    ################
    X_txtsentiment_scr = train_df['text_sentiment_scr'].to_numpy()
    cde = X_txtsentiment_scr.reshape(-1, 1)

    X_glv = train_df['glove_vec'].to_numpy()
    X_vec = train_df['vec'].to_numpy()
    X_claimspadpoldiffvec = train_df['claims_padded_polaritydiff_vec'].to_numpy()

    abc = X_vec.reshape(-1, 1)
    abc = np.concatenate(np.concatenate(abc, axis = 0), axis = 0).reshape(-1, 300)
    pqr = np.concatenate(X_glv, axis = 0).reshape(-1, 100)
    rst = np.concatenate(X_claimspadpoldiffvec, axis = 0).reshape(-1, 9)

    X_train = np.hstack([cde,abc,pqr,rst,uvw])
    y_train = train_df['confirmation_bias']
    
    ################
    X_txtsentiment_scr_test = test_df['text_sentiment_scr'].to_numpy()
    cde_test = X_txtsentiment_scr_test.reshape(-1, 1)

    X_glv_test = test_df['glove_vec'].to_numpy()
    X_vec_test = test_df['vec'].to_numpy()
    X_claimspadpoldiffvec_test = test_df['claims_padded_polaritydiff_vec'].to_numpy()

    abc_test = X_vec_test.reshape(-1, 1)
    abc_test = np.concatenate(np.concatenate(abc_test, axis = 0), axis = 0).reshape(-1, 300)

    pqr_test = np.concatenate(X_glv_test, axis = 0).reshape(-1, 100)

    rst_test = np.concatenate(X_claimspadpoldiffvec_test, axis = 0).reshape(-1, 9)

    X_test = np.hstack([cde_test,abc_test,pqr_test,rst_test,uvw_test])
    y_test = test_df['confirmation_bias']

    ################
    # clf = SVC(kernel = 'rbf', C = 1000, gamma='auto').fit(X_train, y_train)
    # clf = RandomForestClassifier(max_features=None, n_jobs=-1)
    # clf.fit(mining_prepared, y_train)

    pipe = Pipeline([
        ('clf', SVC()),
    ])
    hyperparameters = {
        'clf__kernel': ('rbf','linear','sigmoid'),
        'clf__degree': (2,3,4,5),
        'clf__C': (1,5,10,100,500,1000)
    }
    clf = GridSearchCV(pipe, hyperparameters, n_jobs=-1, cv = 5)
    clf.fit(X_train, y_train)
    print(clf.best_estimator_)
    print(clf.best_params_)
    print(clf.best_score_)

    y_pred = clf.predict(X_test)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f1_macro)
    print(f1_weighted)

    predictions_file = []
    for x in zip(test_df['id'], y_pred):
        opt = {
            "id": int(x[0]),
            "confirmation_bias": bool(x[1])
        }
        predictions_file.append(opt)
    
    json.dump(predictions_file, open('./predictions.json','w'))

    predictions_file_df = pd.DataFrame(predictions_file)
    predictions_file_df.to_csv('./predictions.csv', header=False, index=False)
    
    print("it works!")
    pass

if __name__ == '__main__':
    main()