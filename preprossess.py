import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from math import sin, cos, sqrt, atan2, radians
from sklearn.preprocessing import StandardScaler
import os

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction import DictVectorizer

df = pd.read_csv("zomato.csv", encoding='latin-1', header=None)

df.columns = ['RestaurantID', 'RestaurantName', 'CountryCode', 'City', 'Address', 'Locality', 'LocalityVerbose',
              'Longitude', 'Latitude', 'Cuisines', 'AverageCostfortwo', 'Currency', 'HasTablebooking',
              'HasOnlinedelivery', 'Isdeliveringnow', 'Switchtoordermenu', 'Pricerange', 'Aggregaterating',
              'Ratingcolor', 'Ratingtext', 'Votes']

df.dropna()

temp_types = df.ix[:,'Cuisines'].values

categories = []

for i in temp_types:
        a = str(i).split(', ')
        for j in a:
            if j not in categories:
                categories.append(j)

# print(categories)






# print(df.shape)
# for i in df.head(0):
#     print(i)

#
lat = df.ix[:, 7:8].values
lon = df.ix[:, 8:9].values
distFromCenter = []

lat1, lon1 = 51.495105, -0.112788


def distancefromCenter(lat, lon):
    R = 6371
    dLat = radians(lat - lat1)
    dLon = radians(lon - lon1)
    a = sin(dLat / 2) * sin(dLat / 2) + cos(radians(lat)) * cos(radians(0)) * sin(dLon / 2) * sin(dLon / 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    d = R * c
    return d


for i in range(len(lat)):
    distFromCenter.append(distancefromCenter(lat[i], lon[i]))
#
# print(pd.Series(np.where(df['HasOnlinedelivery'].values == 'yes', 1, 0), df.index))

w = pd.Series(temp_types).str.get_dummies(', ')

w['HasDelivery'] = pd.Series(np.where(df['HasOnlinedelivery'].values == 'Yes', 1, 0), df.index)

# w['RestaurantName'] = df['RestaurantName']

# w['AverageCost'] = df['AverageCostfortwo']

w['Pricerange'] = df['Pricerange']

w['DistFromCenter'] = StandardScaler().fit(distFromCenter)


w['Ratingtext'] = df['Ratingtext']


w.to_csv('final.csv', sep=',', encoding='utf-8')





# print(df['RestaurantID'])
#
#
#
# cuisine_dict = (df.set_index('RestaurantID')['Cuisines'].to_dict('list'))
# cuisine_dict = dict(zip(df['RestaurantName'], df['Cuisines']))
#
# temp_df = df.head()

# cuisine_dict = df.set_index(['RestaurantName'])['Cuisines'].to_dict()
#
#

#
# main_matrix = []
#
# for i in cuisine_dict:
#     one_line_matrix = [0] * len(categories)
#     a = str(cuisine_dict.get(i)).split(', ')
#     for j in a:
#         print(j)
#         one_line_matrix[categories.index(j)] = 1
#     main_matrix.append(one_line_matrix)
#
# with open('onehot.csv', 'w') as f:
#     for category in categories:
#         f.write("%s," % category)
#     f.write("\n")
#     for item in main_matrix:
#         for number in item:
#             f.write("%s," % number)
#         f.write("\n")
# print(cuisine_dict)
# cuisine_dict1 = cuisine_dict.set_index('temp').to_dict()
# print(df['temp'])

# dict_one_hot = DictVectorizer(sparse= False)
# cuisine_encoded = dict_one_hot.fit_transform(cuisine_dict)
# print(cuisine_encoded)



