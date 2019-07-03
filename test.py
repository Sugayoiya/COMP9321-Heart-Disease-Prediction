from visualise import *
from ml import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def get_important_factors_list():
	df = data_preprocessing()
	# print(df)
	df = convert_binary(df)
	# print(df)
	plot_check_imbalance(df)
	X, y = get_xy(df)
	factor_score = evaluate_factors(X, y)
	names = ["age","sex","chest_pain_type","resting_blood_pressure","serum_cholestoral",\
	    "fasting_blood_sugar","resting_electrocardiographic_results","maximum_heart_rate_achieved","exercise_induced_angina",\
	    "oldpeak","the_slope_of_the_peak_exercise_ST_segment","number_of_major_vessels_colored_by_ourosopy","thal","target"]
	important_factors, important_factors_list = choose_imfactor(factor_score, names)
	##important_factors_list  contains the indexes of important factors
	important_factors_list_ = [i+1 for i in important_factors_list]
	return important_factors_list_

def predict_model():

	df = data_preprocessing()
	# print(df)
	df = convert_binary(df)
	# print(df)
	plot_check_imbalance(df)
	X, y = get_xy(df)
	factor_score = evaluate_factors(X, y)
	names = ["age","sex","chest_pain_type","resting_blood_pressure","serum_cholestoral",\
	    "fasting_blood_sugar","resting_electrocardiographic_results","maximum_heart_rate_achieved","exercise_induced_angina",\
	    "oldpeak","the_slope_of_the_peak_exercise_ST_segment","number_of_major_vessels_colored_by_ourosopy","thal","target"]
	plot_important_factors(factor_score)
	important_factors, important_factors_list = choose_imfactor(factor_score, names)
	##important_factors_list  contains the indexes of important factors
	print('important_factors_list:',important_factors_list)
	#start logistic regression
	X, original_X = get_newX(X, important_factors)

	# normalization_columns = ["age", "chest_pain_type","resting_blood_pressure","serum_cholestoral","maximum_heart_rate_achieved","oldpeak","thal"]
	# max_data_list, min_data_list = normalize(X, normalization_columns)

	#run PCA on data to further increase accuracy
	# X = pca_analyze(X)

	iterations_vs_accuracy, iterations_vs_accuracy_cv, X_test, y_test, y_pred, lr = train(X,y)

	#plot confusion matrix
	plot_conmatrix(y_test, y_pred)

	#plot iterations vs accuracy
	title_name = 'Iterations vs Accuracy'
	y_label = 'Accuracy'
	figname = "iterations_vs_accuracy.png"
	plot_ia(iterations_vs_accuracy, title_name, y_label, figname)

	title_name = 'Iterations vs Accuracy (K-fold Cross Validation)'
	y_label = 'Accuracy (K-fold Cross Validation)'
	figname = "iterations_vs_accuracy_cv.png"
	plot_ia(iterations_vs_accuracy_cv, title_name, y_label, figname)

	#plot roc graph
	plot_roc_curve(y_test,X_test,lr)

	#save model
	filename = 'lr.model'
	save_model(lr, filename)

	#cross validation
	loaded_model, kfold_score = cv_model(original_X, y, filename)
	print('kfold_score:',kfold_score)
	a = round(kfold_score,2)
	return [loaded_model,names,important_factors_list,original_X,a]

#predict
def predict_user_input(input_data):
	data = predict_model()
	# print(data)
	# input_data = [57.0,2.0,130.0,236.0,174.0,0.0,0.0,1.0,3.0]
	if 0 == predict(data[0], input_data, data[1], data[2], data[3]):
		return 'NO'
	else:
		return 'YES'
