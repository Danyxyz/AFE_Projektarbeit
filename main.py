from pandas.core.dtypes.inference import Number
from pandas.core.common import random_state
from pandas import DataFrame
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from numpy import savetxt
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# read in the data, only use relevant coloumns
data = pd.read_csv("Bauzonen_Gebaeude_2.csv", on_bad_lines="warn", engine="python", delimiter=',',
                   usecols=['main_gemeinde_data_Index', 'CH_BEZ_D', 'GKODE', 'GKODN'])

# Formatting, putting in a DataFrame
df_gebaude = pd.DataFrame(data)
df_gebaude = df_gebaude.rename(columns={'CH_BEZ_D': 'Bauzone'})
df_gebaude = df_gebaude.rename(columns={'main_gemeinde_data_Index': 'Index'})

# Read and format local communities
gemeindeData = pd.read_csv("main_gemeinde_data.csv", engine="python", delimiter=';',
                           usecols=['Index', 'Gemeinde_Nr', 'K1_Hoch', 'K2_Mittel', 'K3_Tief'])
df_gemeindeData = pd.DataFrame(gemeindeData)
print("Data successfully read")

low_zone_NORTH = []
low_zone_EAST = []
middle_zone_NORTH = []
middle_zone_EAST = []
high_zone_NORTH = []
high_zone_EAST = []
qgis_coordinate_list = []



#Loop through community set and work with subsets of each community
#Community table is modified due to wrong bfs-number on original document, working with indeces
#for i in range(0, len(gemeindeData) -1):
#Test Set for Luzern (7500 Datapoints)
for i in range(0, len(gemeindeData)-1):
    df_subset = df_gebaude[df_gebaude.Index == i]
    k_iterator = i
    print("Gemeindefortschritt ", i)

#Loop through the subsets and store each zones in different list, Low / Middle / High
    for i in range(0, len(df_subset)-1):
        if df_subset.Bauzone.iloc[i] == 'Wohnzonen' or df_subset.Bauzone.iloc[i] == 'eingeschränkte Bauzonen' or df_subset.Bauzone.iloc[i] == 'weitere Bauzonen' or df_subset.Bauzone.iloc[i] == 'Verkehrszonen innerhalb der Bauzonen':
            low_zone_NORTH.append(df_subset.GKODN.iloc[i])
            low_zone_EAST.append(df_subset.GKODE.iloc[i])
        else:
            if df_subset.Bauzone.iloc[i] == 'Arbeitszonen' or df_subset.Bauzone.iloc[i] == 'Mischzonen':
                middle_zone_NORTH.append(df_subset.GKODN.iloc[i])
                middle_zone_EAST.append(df_subset.GKODE.iloc[i])
            else:
                high_zone_NORTH.append(df_subset.GKODN.iloc[i])
                high_zone_EAST.append(df_subset.GKODE.iloc[i])

#Putting lists in dataframes to make process easier
    df_low_zone_NORTH = pd.DataFrame(low_zone_NORTH)
    df_low_zone_EAST = pd.DataFrame(low_zone_EAST)
    df_middle_zone_NORTH = pd.DataFrame(middle_zone_NORTH)
    df_middle_zone_EAST = pd.DataFrame(middle_zone_EAST)
    df_high_zone_NORTH = pd.DataFrame(high_zone_NORTH)
    df_high_zone_EAST = pd.DataFrame(high_zone_EAST)

#Merging x and y coordinates to one list
    df_low = pd.merge(df_low_zone_NORTH, df_low_zone_EAST, left_index=True, right_index=True)
    df_middle = pd.merge(df_middle_zone_NORTH, df_middle_zone_EAST, left_index=True, right_index=True)
    df_high = pd.merge(df_high_zone_NORTH, df_high_zone_EAST, left_index=True, right_index=True)

#numpy arrays instead of dataframe - much faster and kmeans requiers np
    np_array_low = df_low.to_numpy()
    np_array_middle = df_middle.to_numpy()
    np_array_high = df_high.to_numpy()


    # Begin kmeans Calculations with low
    # Some communities might not have enough zones to calculate a proper kmeans clustering hence try

#--------------------------------------------------------------------------------------------------------kmeans on first lists low

    k = df_gemeindeData.K3_Tief[k_iterator]
    k = k.astype(int)
    k = round(k * 0.2)

    #trying to solve problem of having not enough data on buildings zones for small communities
    if (np_array_low.size / 2 < k and np_array_low.size != 0):
        N = 2
        while np_array_low.size / 2 < k:
            np_array_low = np.vstack([np_array_low] * N)

    try:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(np_array_low)
        # Set the labels
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        # Store centroids of x and y individually
        centroids_x = centroids[:, 0]
        centroids_y = centroids[:, 1]

        coordinate_list = []
        for i in range(len(centroids_x)):
            number = np.round(centroids[i], 2)
            coordinate_list.append(number)
        # Append to list everything to list before final saving
    except Exception:
        pass

    finally:
        qgis_coordinate_list.extend(coordinate_list)
        k = df_gemeindeData.K3_Tief[k_iterator]
        k = k.astype(int)
        k = round(k * 0.4)

        if (np_array_middle.size / 2 < k and np_array_middle.size != 0):
            N = 2
            while np_array_middle.size / 2 < k:
                np_array_middle = np.vstack([np_array_middle] * N)
    try:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(np_array_middle)
        # Set the labels
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        # Store centroids of x and y individually
        centroids_x = centroids[:, 0]
        centroids_y = centroids[:, 1]

        coordinate_list = []
        for i in range(len(centroids_x)):
            number = np.round(centroids[i], 2)
            coordinate_list.append(number)

        qgis_coordinate_list.extend(coordinate_list)
    except Exception:
        pass
    finally:
#-------------------------------------------------------------------------------Run kmeans on third list high
        k = df_gemeindeData.K3_Tief[k_iterator]
        k = k.astype(int)
        k = round(k * 0.4)


        if (np_array_high.size / 2 < k and np_array_high.size != 0):
            N = 2
            while np_array_high.size / 2 < k:
                np_array_high = np.vstack([np_array_high] * N)

    try:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(np_array_high)
        # Set the labels
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        # Store centroids of x and y individually
        centroids_x = centroids[:, 0]
        centroids_y = centroids[:, 1]

        coordinate_list = []
        for i in range(len(centroids_x)):
            number = np.round(centroids[i], 2)
            coordinate_list.append(number)
    except Exception:
        pass
    finally:
        qgis_coordinate_list.extend(coordinate_list)
        k_iterator = k_iterator + 1

#save to csv and export
np.savetxt("Charging_Stations_K3_Tief.csv", qgis_coordinate_list, delimiter=",", fmt="%f")
#np.savetxt("Charging_Stations.csv", qgis_coordinate_list, delimiter=",", fmt="%f")
#Number should be around half a million records
print("Ladestationen ", len(qgis_coordinate_list))
print("Done")
