import sys
import pandas as pd
import numpy as np
import math
from math import cos, asin, sqrt
import matplotlib.pyplot as plt

#python3 temperature_correlation.py stations.json.gz city_data.csv output.svg

stations_file = sys.argv[1]
city_file = sys.argv[2]
output = sys.argv[3]
'''
stations = pd.read_json(stations_file, lines=True)
city_data = pd.read_csv(city_file) 
'''
stations = pd.read_json(stations_file, lines=True)
city_data = pd.read_csv(city_file) 
#print(stations)
#print(city_data)

# THE DATA
stations['avg_tmax'] = stations['avg_tmax'] / 10

#Inspired from https://www.kite.com/python/answers/how-to-drop-empty-rows-from-a-pandas-dataframe-in-python#:~:text=Use%20df.,contain%20NaN%20under%20those%20columns.
city_data.dropna(subset = ['population'], inplace = True)
city_data.dropna(subset = ['area'], inplace = True)
#print(city_data)

city_data['area'] = city_data['area'] / 1000000 # one m^2 is 0.000001 km^2
city_data = city_data[city_data['area'] <= 10000]

#print(city_data)
#print(stations)

def distance(city_data, stations):
    # calculates the distance between one city and every station
    lat1, lon1 = city_data.loc['latitude'], city_data.loc['longitude']
    lat2, lon2 = stations['latitude'].values, stations['longitude'].values
    return distanceBetweenPoints(lat1, lon1, lat2, lon2)

def distanceBetweenPoints(lat1, lon1, lat2, lon2): # From my exercise 3
    p = math.pi/180
    a = 0.5 - np.cos((lat2-lat1)*p)/2+np.cos(lat1*p)*np.cos(lat2*p)*(1-np.cos((lon2-lon1)*p))/2
    return 12742 * np.arcsin(np.sqrt(a))

def best_tmax(city_data, stations):
    dist_to_station = distance(city_data, stations)
    return stations.loc[np.argmin(dist_to_station)]['avg_tmax']


city_data['temperature'] = city_data.apply(best_tmax, axis=1, stations=stations)
city_data['density'] = city_data['population'] / city_data['area']

# Scatterplot
plt.scatter(city_data['temperature'], city_data['density'])
plt.title('Temperature vs Population Density')
plt.xlabel('Avg Max Temperature (\u00b0C)')
plt.ylabel('Population Density (people/km\u00b2)')
plt.savefig(output)