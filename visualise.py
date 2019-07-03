import matplotlib.pyplot as plt
import pandas as pd
import os

def data_preprocessing():
	csv_file = 'processed.cleveland.data'
	df = pd.read_csv(csv_file,header=None,names=["age","sex","chest_pain_type","resting_blood_pressure","serum_cholestoral","fasting_blood_sugar","resting_electrocardiographic_results","maximum_heart_rate_achieved","exercise_induced_angina","oldpeak","the_slope_of_the_peak_exercise_ST_segment","number_of_major_vessels_colored_by_ourosopy","thal","target"])

	#clean data
	for header in df:
		df =df[~df[header].isin(["?"])]
	df = df.apply(pd.to_numeric)
	return df


def visualisation():
	df = data_preprocessing()
	#mkdir
	if not os.path.exists("static/pic"):
		os.mkdir("static/pic")

	#set lable
	male = df.query('sex == 1')
	female = df.query('sex == 0')

	#3. chest pain type (1=typical angin,2=atypical angina,3=non-anginal pain,4=asymptomatic)
	chest_pain_type = male.plot.scatter(x='chest_pain_type', y='age', label='male')
	chest_pain_type = female.plot.scatter(x='chest_pain_type', y='age', label='female',color="g", ax=chest_pain_type)
	plt.xticks([1,2,3,4])
	plt.title("1=typical angin, 2=atypical angina\n3=non-anginal pain, 4=asymptomatic")
	plt.savefig('static/pic/chest_pain_type.png',dpi=125)

	#4. resting blood pressure
	resting_blood_pressure = male.plot.scatter(x='resting_blood_pressure', y='age', label='male')
	resting_blood_pressure = female.plot.scatter(x='resting_blood_pressure', y='age', label='female', color="g", ax=resting_blood_pressure)
	plt.savefig('static/pic/resting_blood_pressure.png',dpi=125)

	#5. serum cholestoral in mg/dl
	serum_cholestoral = male.plot.scatter(x='serum_cholestoral', y='age', label='male')
	serum_cholestoral = female.plot.scatter(x='serum_cholestoral', y='age', label='female', color="g", ax=serum_cholestoral)
	plt.savefig('static/pic/serum_cholestoral.png',dpi=125)

	#6. fasting blood sugar > 120 mg/dl
	fasting_blood_sugar = male.plot.scatter(x='fasting_blood_sugar', y='age', label='male')
	fasting_blood_sugar = female.plot.scatter(x='fasting_blood_sugar', y='age', label='female', color="g", ax=fasting_blood_sugar)
	plt.xticks([0,1])
	plt.title("0=fasting blood sugar <= 120 mg/dl\n1=fasting blood sugar > 120 mg/dl")
	plt.savefig('static/pic/fasting_blood_sugar.png',dpi=125)

	#7. resting electrocardiographic results (0=normal,1=having ST-T wave abnormality (T wave in-versions and/or ST elevation or depression of > 0.05 mV),2=showing probable or denite leftventricular hypertrophy by Estes' criteria)
	resting_electrocardiographic_results = male.plot.scatter(x='resting_electrocardiographic_results', y='age', label='male')
	resting_electrocardiographic_results = female.plot.scatter(x='resting_electrocardiographic_results', y='age', label='female', color="g", ax=resting_electrocardiographic_results)	
	plt.xticks([0,1,2])
	plt.title("0=normal, 1=having ST-T wave abnormality\n2=showing probable or denite leftventricular\nhypertrophy by Estes' criteria")
	plt.tight_layout()
	plt.savefig('static/pic/resting_electrocardiographic_results.png',dpi=125)

	#8. maximum heart rate achieved
	maximum_heart_rate_achieved = male.plot.scatter(x='maximum_heart_rate_achieved', y='age', label='male')
	maximum_heart_rate_achieved = female.plot.scatter(x='maximum_heart_rate_achieved', y='age', label='female', color="g", ax=maximum_heart_rate_achieved)
	plt.savefig('static/pic/maximum_heart_rate_achieved.png',dpi=125)

	#9. exercise induced angina
	exercise_induced_angina = male.plot.scatter(x='exercise_induced_angina', y='age', label='male')
	exercise_induced_angina = female.plot.scatter(x='exercise_induced_angina', y='age', label='female', color="g", ax=exercise_induced_angina)
	plt.xticks([0,1])
	plt.savefig('static/pic/exercise_induced_angina.png',dpi=125)

	#10. oldpeak = ST depression induced by exercise relative to rest
	oldpeak = male.plot.scatter(x='oldpeak', y='age', label='male')
	oldpeak = female.plot.scatter(x='oldpeak', y='age', label='female', color="g", ax=oldpeak)
	plt.savefig('static/pic/oldpeak.png',dpi=125)

	#11. the slope of the peak exercise ST segment
	the_slope_of_the_peak_exercise_ST_segment = male.plot.scatter(x='the_slope_of_the_peak_exercise_ST_segment', y='age', label='male')
	the_slope_of_the_peak_exercise_ST_segment = female.plot.scatter(x='the_slope_of_the_peak_exercise_ST_segment', y='age', label='female', color="g", ax=the_slope_of_the_peak_exercise_ST_segment)
	plt.xticks([1,2,3])
	plt.savefig('static/pic/the_slope_of_the_peak_exercise_ST_segment.png',dpi=125)

	#12. number of major vessels (0-3) colored by flourosopy
	number_of_major_vessels_colored_by_ourosopy = male.plot.scatter(x='number_of_major_vessels_colored_by_ourosopy', y='age', label='male')
	number_of_major_vessels_colored_by_ourosopy = female.plot.scatter(x='number_of_major_vessels_colored_by_ourosopy', y='age', label='female', color="g", ax=number_of_major_vessels_colored_by_ourosopy)
	plt.xticks([0,1,2,3])
	plt.savefig('static/pic/number_of_major_vessels_colored_by_ourosopy.png',dpi=125)

	#13. thal(Thalassemia): 3 = normal; 6 = fixed defect; 7 = reversable defect
	thal = male.plot.scatter(x='thal', y='age', label='male')
	thal = female.plot.scatter(x='thal', y='age', label='female', color="g", ax=thal)
	plt.xticks([3,6,7])
	plt.title("3 = normal, 6 = fixed defect\n7 = reversable defect")
	plt.savefig('static/pic/thal.png',dpi=125)

	#plt.show()

