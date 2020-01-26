from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle


def storeData(obj, file):
    saveFile = open(file, 'ab')

    pickle.dump(obj, saveFile)
    saveFile.close()

def encode(district_name):
    global code

    if district_name in district_name_to_code_map:
        return district_name_to_code_map[district_name]
    else:
        code = code + 1
        district_name_to_code_map[district_name] = code
        return code

def applyDistrictEncoding(df):
    for i in range(len(df.index)):
        df.iat[i, 1] = encode(df.iat[i, 1])

code = 0
district_name_to_code_map = {}
table = pd.read_csv('dataset_crop.csv')

applyDistrictEncoding(table)

feature_names = ['District_Name', 'Temperature', 'Rainfall']
X = table[feature_names]
y = table['Crop']

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

knn = KNeighborsClassifier()
knn.fit(X, y)

storeData(district_name_to_code_map, 'district_names_map')
storeData(scaler.data_range_, 'data_range')
storeData(scaler.data_min_, 'data_min')
storeData(y, 'crop_labels')
storeData(knn, 'knn_classifier')
