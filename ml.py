import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

'''
=========plot about data, before training============================
'''

#check the data is imbalance or not
def plot_check_imbalance(df):
    imbalance = sns.countplot(x='target',data=df, palette='Set1')
    plt.title('Inspection of Target Data (Check for imbalance)')
    fig1 = imbalance.get_figure()
    fig1.savefig("static/predict/check_imbalance.png")
    

#plot important factors graph
def plot_important_factors(score):
    score.plot(x='Attributes',y='Score', kind='barh',figsize=(15, 7) )
    plt.title('Score of Each Attributes')
    plt.xlabel('Attibutes')
    plt.ylabel('Score')
    plt.yticks(fontsize=10)
    plt.subplots_adjust(left=0.3)
    plt.savefig("static/predict/important_factors.png")


'''
=========plot of evaluating algrithm performance, after training===========
'''

#plot roc graph
def plot_roc_curve(y_test,X_test,lr):
    auc = roc_auc_score(y_test, lr.predict(X_test))
    fpr, tpr, _ = roc_curve(y_test, lr.predict_proba(X_test)[:,1])
    roc = plt.figure()
    plt.plot(fpr, tpr, label=' LogisticRegression (area = %0.2f)' % auc)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.savefig("static/predict/roc.png")

#plot confusion matrix
def plot_conmatrix(y_test, y_pred):
    con_m = confusion_matrix(y_test, y_pred)
    cm = pd.DataFrame(con_m, index = [i for i in "01"], columns = [i for i in "01"])
    cm_fig = plt.figure()
    heatmap = sns.heatmap(cm, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("static/predict/confusion_matrix.png")

#plot iterations vs accuracy
def plot_ia(iterations_vs_accuracy, title_name, y_label, figname):
    iterations_vs_accuracy.plot(x='num_iterations',y='accuracy',legend=None)
    plt.title(title_name)
    plt.xlabel('Number of Iterations')
    plt.ylabel(y_label)
    plt.savefig('static/predict/'+figname)


### fucntions to improve accuracy of model ###
#normalization of data
# def normalize(X, normalization_columns):

#     max_data_list = []
#     min_data_list = []
#     for x in X.columns:
#         if x in normalization_columns:
#             max_data = X[x].max()
#             min_data = X[x].min()
#             max_data_list.append(max_data)
#             min_data_list.append(min_data)
#             X[x] = (X[x] - min_data) / (max_data - min_data)
#     return max_data_list, min_data_list

#PCA used to reduce dimensions
def pca_analyze(X):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.9, svd_solver='full')
    X = pca.fit_transform(X)
    return X

#cane normalize to standarlize
def standarlize_pca(X_train, X_test):
    # from sklearn.preprocessing import MinMaxScaler
    # from sklearn.decomposition import PCA
    
    # min_max_scaler = MinMaxScaler()
    # X_train_minmax = min_max_scaler.fit_transform(X_train)
    # X_test_minmax = min_max_scaler.transform(X_test)

    scaler_train = StandardScaler().fit(X_train)
    scaled_train = scaler_train.transform(X_train)
    scaled_test = scaler_train.transform(X_test)
    
    # pca = PCA(n_components=4, svd_solver='full')
    # X_train_pca = pca.fit_transform(scaled_train)
    # X_test_pca = pca.transform(scaled_test)
    # print(pca.explained_variance_ratio_)
    # print(X_train_pca)
    # return X_train_pca, X_test_pca
    return scaled_train, scaled_test
    # return X_train_minmax, X_test_minmax

## deal with initial data ##
#convert target column data to 1 and 0
def convert_binary(df):
    df['target'] = df['target'].apply(lambda x: 1 if x >=1 else 0)
    return df

#get corresponding X and y
def get_xy(df):
    X = df.iloc[:,0:13]  #independent columns
    y = df.iloc[:,-1]    #target column
    return X, y

#make X only have important factors
def get_newX(X, important_factors):
    X = X[important_factors]
    original_X = X.copy()
    return X, original_X

## function to distinguish the important factors ##
#SelectKBest chi-squared class used to list out the more important attributes
def evaluate_factors(X, y):
    attributes = SelectKBest(score_func=chi2, k=13)
    res = attributes.fit(X,y)
    points = pd.DataFrame(res.scores_)
    col = pd.DataFrame(X.columns)

    #concat both dataframes for visual representation 
    attributes_scores = pd.concat([col,points],axis=1)
    attributes_scores.columns = ['Attributes','Score']
    attributes_vs_score = attributes_scores.nlargest(13,'Score')
    return attributes_vs_score

#choose only the important attributes based on score
def choose_imfactor(attributes_vs_score, names):
    #the variable acceptable_score is the cutoff point 
    #(if score is lower then that attribute is dropped)
    acceptable_score = 10
    important_factors = []
    for index, row in attributes_vs_score.iterrows():
        if row['Score'] >= acceptable_score:
            important_factors.append(row['Attributes'])
        else:
            break

    important_factors_list = sorted([names.index(i) for i in important_factors])
    return important_factors, important_factors_list


#logistic regression used to obtain model which will allow for heart disease prediction
#number of iteration is limited to 100 as model is shown to converge before that epoch
def train(X, y):
    iterations_vs_accuracy = pd.DataFrame(columns = ['num_iterations', 'accuracy'])
    iterations_vs_accuracy_cv = pd.DataFrame(columns = ['num_iterations', 'accuracy'])

    for num_iter in range(0,101,5):
        #split train and test set to 75% and 25% respectively for logistic regression
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=71)
        X_train, X_test = standarlize_pca(X_train, X_test)
        lr = LogisticRegression(penalty='l2', solver="lbfgs", max_iter=num_iter)
        kfold_scores = cross_val_score(lr,X,y,cv=5).mean()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        acc = np.mean(y_pred == y_test)
        #save accuracy for each iteration in dataframe for graph plotting
        iterations_vs_accuracy = iterations_vs_accuracy.append({
            'num_iterations': num_iter,
            'accuracy': acc
            }, ignore_index = True)
        # print(f"num_iterations: {num_iter}    accuracy: {acc}")
        iterations_vs_accuracy_cv = iterations_vs_accuracy_cv.append({
            'num_iterations': num_iter,
            'accuracy': kfold_scores
            }, ignore_index = True)
    return iterations_vs_accuracy, iterations_vs_accuracy_cv, X_test, y_test, y_pred, lr



#finish training
#save model
def save_model(lr, filename='lr.model'):

    pickle.dump(lr, open(filename, 'wb'))

#cross validation
def cv_model(X, y, filename='lr.model'):
    loaded_model = pickle.load(open(filename, 'rb'))
    kfold_scores = cross_val_score(loaded_model,X,y,cv=5).mean()
    return loaded_model, kfold_scores


#predict based on user input
def predict(loaded_model, input_data, names, important_factors_list, original_X): 
    inputs = pd.DataFrame(np.array([input_data]), columns = [names[i] for i in important_factors_list])

    original_X = original_X.head(100).append(inputs, ignore_index=True)
    scaler_train = StandardScaler().fit(original_X)
    scaled_train = scaler_train.transform(original_X)

    result = loaded_model.predict(original_X)[-1]
    print(result)
    return result 

#random forest
# from sklearn.ensemble import RandomForestClassifier
# fitRF = RandomForestClassifier(random_state = 71, 
#                                 criterion='gini',
#                                 n_estimators = 500,
#                                 max_features = 4)
# # print(y_train)
# kfold_scores = cross_val_score(fitRF,X,y,cv=5)
# print(kfold_scores)
# print("Accuracy: %0.2f (+/- %0.2f)" % (kfold_scores.mean(), kfold_scores.std() * 2))
# fitRF.fit(X_train, y_train['target'])
# importancesRF = fitRF.feature_importances_
# indicesRF = np.argsort(importancesRF)[::-1]
# predictions_RF = fitRF.predict(X_test)
# # print(predictions_RF)
# # print(y_test['target'])
# print(len(predictions_RF))
# p = np.mean(predictions_RF == y_test['target'])
# print(p)

# # print(predictions_RF)
# # print(y_test['target'])
# accuracy_RF = fitRF.score(X_test, y_test)
# print(accuracy_RF)
