import json
import pandas as pd
import gensim 
import numpy as np
import nltk
import spacy
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize,sent_tokenize
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from spacytextblob.spacytextblob import SpacyTextBlob
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')
from sklearn.metrics import accuracy_score,f1_score
from sklearn.ensemble import RandomForestClassifier

def main():
    ESSAYS_PATH = './data/essay-corpus.json'
    with open(ESSAYS_PATH, "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    TRAINTESTSPLITPATH = './data/train-test-split.csv'
    trainTestSplit = pd.read_csv(TRAINTESTSPLITPATH,sep=';')
    
    model = buildWord2Vec(sent_tokenize(' '.join(df['text'])))
    
    # Contextual Features
    df['topic'] = df['text'].apply(lambda x: np.array(sent2vec(model,x.split('\n')[0])))
    df['all_major_claim'] = extratAll(model,df['major_claim'])
    df['all_premises'] = extratAll(model,df['premises'])
    df['conclusion'] = df['text'].apply(lambda x: np.array(sent2vec(model,x.split('\n')[-1])))
    df['sentiment'] = df['text'].apply(lambda x: nlp(x)._.blob.polarity)
    
    # Split
    df['set'] = trainTestSplit['SET'].values    
    train_df = df[df['set'] == "TRAIN"]
    test_df = df[df['set'] == "TEST"]
        
    # Train n test features 
    train_x = train_df[['topic','all_major_claim','all_premises','conclusion','sentiment']]
    train_y = train_df['confirmation_bias']
    test_x = test_df[['topic','all_major_claim','all_premises','conclusion','sentiment']]
    test_y = test_df['confirmation_bias']

    train_x = train_x.fillna(0)
    test_x = test_x.fillna(0)

    predictions = SVM(train_x,test_x,train_y,test_y)
    
    # Generate output
    generatePredictionJSON(predictions,test_df['id'].reset_index(drop=True))
    
    print("File prediction.json generated succefully.")
    # End main
    
def generatePredictionJSON(predictions,ids):
    output = {
        'id': [],
        'confirmation_bias': []
    }
    for p in range(len(predictions)):
        output['id'].append(str(ids[p]))
        output['confirmation_bias'].append(str(predictions[p]))
    
    o = pd.DataFrame(output)
    o.to_json('prediction.json',orient='records')

def SVM(x_train,x_test,y_train,y_test):
    model = svm.SVC(C=0.1,gamma=1,kernel='rbf')
    # param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'linear','sigmoid']}
    # params = {
    #     'model': model,
    #     'param_grid': param_grid,
    #     'cv': 10,
    #     'x_train':x_train ,'x_test':x_test,'y_train':y_train,'y_test':y_test
    # }
    # hyperParameterTuningAndCV(params) C=0.1,gamma=1,kernel='rbf'
    model.fit(x_train, y_train)
    # 10fold cross validation with polynomial as kernel type.
    scores = cross_val_score(model, x, y, cv=10)
    print(scores)
    print("SVM F1 score(mean): %.3f " % (mean(scores)))
    model.fit(x_train,y_train)
    predictions = model.predict(x_test)
    return predictions

def buildWord2Vec(sentences):
    model = gensim.models.Word2Vec(min_count=1,window=10,vector_size=1)
    corpus_iterable = [word_tokenize(i) for i in sentences]
    model.build_vocab(corpus_iterable=corpus_iterable)
    model.train(corpus_iterable=corpus_iterable,total_examples=len(corpus_iterable),epochs=model.epochs)
    # model.save('wor2vec.model')
    # model = gensim.models.Word2Vec.load('wor2vec.model')
    return model

def word2vec(model,word):
    try:
        return model.wv[word]
    except KeyError:
        default_vector = [0.00000000] 
        return default_vector

def sent2vec(model,sent):
    words = word_tokenize(sent)
    v = []
    for w in words:
        try:
            v.append(word2vec(model,w))
        except:
            continue
    return np.array(v).mean()

def GNB(x_train,x_test,y_train,y_test):
    model = GaussianNB()
    # param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}
    # params = {
    #     'model': model,
    #     'param_grid': param_grid,
    #     'cv': 10,
    #     'x_train':x_train ,'x_test':x_test,'y_train':y_train,'y_test':y_test
    # }
    # hyperParameterTuningAndCV(params)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print("Accuracy Score -> ",accuracy_score(predictions, y_test)*100)
    print("\nreport: ",f1_score(y_test,predictions))
    return predictions

def RF(x_train,x_test,y_train,y_test):
    model = RandomForestClassifier()
    # param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}
    # params = {
    #     'model': model,
    #     'param_grid': param_grid,
    #     'cv': 10,
    #     'x_train':x_train ,'x_test':x_test,'y_train':y_train,'y_test':y_test
    # }
    # hyperParameterTuningAndCV(params)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print("RF Accuracy Score -> ",accuracy_score(predictions, y_test)*100)
    print("\nreport: ",f1_score(y_test,predictions))
    return predictions

def hyperParameterTuningAndCV(params):
    model = GridSearchCV(estimator=params['model'], param_grid=params['param_grid'], cv=params['cv'])
    model.fit(params['x_train'],params['y_train'])
    predictions = model.predict(params['x_test'])
    print("Best params: ",model.best_params_) 
    return predictions

def extratAll(model,column):
    all = []
    for r in column: 
        text = ''
        for l in r:
            text +=  l['text'] + ' '
        all.append(np.array(sent2vec(model,text)))
    return all

    
if __name__ == '__main__':
    main()
