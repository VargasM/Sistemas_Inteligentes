import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utils import *
'''
Function load data
    This function is used to load the dataset, we are using different sources, in this function we concat this datasets into one
Parameters:
    None
Return 
    Data: DataSet
'''

def load_data ():
    path_to_file = './data/heart_attack_prediction.csv'
    path_to_file1 = './data/heart_cleveland_upload.csv'
    path_to_file2 = './data/heart_disease_dataset.csv'
    path_to_file3 = './data/Heart_Disease_Prediction.csv'
    path_to_file4 = './data/heart_disease_UCI.csv'
    path_to_file5 = './data/heart_failure_clinical_records_dataset.csv'
    #getting data
    X = pd.read_csv(path_to_file)
    X1 = pd.read_csv(path_to_file1)
    X2 = pd.read_csv(path_to_file1)
    X3 = pd.read_csv(path_to_file1)
    X4 = pd.read_csv(path_to_file1)
    X5 = pd.read_csv(path_to_file1)
    
    data = pd.concat([X1, X2, X3, X4, X5])

    print(list(data))
    print(data.head(10))
    print(data.info())
    #No missing values
    print(data.isnull().sum())
    return data

'''
Function exploring_distribution
    This function is used to explore the dataset start exploring age distribution and also the sex
Parameters:
     data: Dataset
Return:
    None
'''
def exploring_distribution(data):
    age_dist_plt = plt.figure(figsize=(15,8))
    sns.displot(data.age,color='#86bf91')
    plt.show() 

    labels=['Male','Female']
    plt.figure(figsize=(6,6))
    plt.pie(data.sex.value_counts(), labels=labels, autopct='%1.1f%%', shadow=True)
    plt.show()

'''
Function binding_age
    This function is binfing the age data to see how aging afect heart disease
Parameters:
    Data: Dataset
Return:
    None
'''
def binding_age(data):
    bins = np.linspace(data.age.min(), data.age.max(),8)
    bins = bins.astype(int)
    df_age = data.copy()
    df_age['binned'] = pd.cut(df_age['age'],bins=bins)
    print(df_age.head())
    plt.figure(figsize=(15,8))
    sns.countplot(x='binned',data = df_age,edgecolor=None)
    plt.show()
    print(df_age.binned.value_counts())

'''
Function: split_age_category
    It is possible to split patients into 3 groups by age. we can the 
    categorizing patients into young adults (ages 18-35 years),
    middle-aged adults (ages 36-55 years), and older adults (aged older than 55 years).
Parameter: 
    Data: DataSet
Return:
    DataSet categorize
'''

def split_age_category(data):
    bins = [data.age.min(), 35, 55, np.inf]
    labels = ['young','middle','older']
    df_cat = data.copy()
    df_cat['binned'] = pd.cut(df_cat['age'],bins=bins,labels=labels)
    print(df_cat.head())
    print(data.ca.value_counts())
    plotAxesSbubplot(df_cat, 'binned', 'cp', "Binned Age Groups", "CP Percentages")
    return df_cat
 
'''
Function: thalemesia_analysis
    Function to explore thalemesia parameter
Parameter: 
    Data: DataSet
Return:
    Node
'''    
def thalemesia_analysis(data):
    plotAxesSbubplot(data, 'thal', 'condition', "Thalemesia", "Percentages", ['Disease Free','Has Disease'])
    print(data.thal.value_counts())
    
'''
Function: heart_rate
    Function use to define the limit for normal heart rate according to their age category
    then categorizing the heart rate category as Normal or High in thalach_bin
    and grouping df_t to get the counts of the patients in each group
    Normal maximum heart rate differs with the age of the patient. 
    Thus we will be categorizing the heart rates of patients according to their ages.
    Young patients -> normal < 200 < High
    Middle aged patients -> normal < 180 < High
    Older patients -> normal < 170 < high
Parameter: 
    Data: DataSet
Return:
    Node
''' 
def heart_rate(data):
    df_t = data.copy(deep=True)
    df_t.loc[df_t.binned=='young','hr_bin'] = 200
    df_t.loc[df_t.binned=='middle','hr_bin'] = 185
    df_t.loc[df_t.binned=='older','hr_bin'] = 160
    print(df_t.head())
    #categorizing the heart rate category as Normal or High in thalach_bin
    df_t['thalach_bin'] = np.where(df_t.eval("thalach <= hr_bin "), "Normal", "High")
    print(df_t)
    #grouping df_t to get the counts of the patients in each group
    df_thalach = df_t.groupby(['thalach_bin','condition','binned']).count()
    print(df_thalach)
    
'''
Function: thalemesia_analysis
    Function to explore and categorize blood pressure.
    Blood pressure, Low < 80 < Normal < 120 < High
Parameter: 
    Data: DataSet
Return:
    DataSet df_trestbps
'''     
def blood_pressure(data):
    df_trestbps = data.copy()
    df_trestbps['trestbps_bin'] = data.apply(trestbps_bin,axis=1)
    df_trestbps
    plotAxesSbubplot(df_trestbps, 'trestbps_bin', 'condition', "Blood Pressure Bin", "Percentages", ['Disease Free','Has Disease'])

    print(df_trestbps.trestbps_bin.value_counts())
    return df_trestbps


'''
Function: exang
    Function to explore exang parameter.
    this is a indicator for a heart disease since for both type 0 and type 1 difference between sick people and healthy people is high.
Parameter: 
    Data: DataSet
Return:
    Node
'''     
def exang(data):
    plotAxesSbubplot(data, 'exang', 'condition', "Exercise Induced Angina", "Percentages", ['Disease Free','Has Disease'])  
  
def run_analysis():
    data = load_data()
    exploring_distribution(data)
    binding_age(data)
    df=split_age_category(data)
    heart_rate(df)
    df_trestbps = blood_pressure(data)
    exang(df_trestbps)
    return df_trestbps