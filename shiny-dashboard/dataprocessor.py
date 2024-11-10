import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from math import radians, sin, cos, sqrt, atan2


app_dir = Path(__file__).parent

data = pd.read_excel("../lotwize_case.xlsx")


def HOADataProducer(df):
    HOA = df.loc[:, ["resoFacts/hasAssociation", "monthlyHoaFee", "resoFacts/atAGlanceFacts/5/factLabel", "resoFacts/atAGlanceFacts/5/factValue"]]
    HOA["hasHOA"] = HOA["resoFacts/atAGlanceFacts/5/factValue"].str.startswith("$", na=False)
    HOA[["HOAFee", "HOAFeeFrequency"]] = HOA["resoFacts/atAGlanceFacts/5/factValue"].where(HOA["hasHOA"]).str.extract(r'^\$(\d+)\s(\D+)')
    HOA["HOAFee"] = HOA["HOAFee"].astype("float64")
    HOA['AnnualHOAFee'] = HOA.apply(
        lambda row: row['HOAFee'] * 12 if row['hasHOA'] and row['HOAFeeFrequency'] == 'monthly' else (row['HOAFee'] if row['hasHOA'] else 0),
        axis=1
    )
    HOA["hasHOA"] = HOA["hasHOA"].astype("boolean")
    df = pd.concat([df, HOA[["hasHOA", "AnnualHOAFee"]]], axis=1)
    return df


def LayoutDataProducer(df):
    df = df.rename(columns={"resoFacts/bathroomsFull": "fullBathrooms", "resoFacts/bathroomsHalf": "halfBathrooms"})
    df["fullBathrooms"] = df["fullBathrooms"].fillna(0)
    df["halfBathrooms"] = df["halfBathrooms"].fillna(0)
    df["bedrooms"] = df["bedrooms"].fillna(0)
    df["fullBathrooms"] = df["fullBathrooms"].astype("Int64")
    df["halfBathrooms"] = df["halfBathrooms"].astype("Int64")
    df["bedrooms"] = df["bedrooms"].astype("Int64")
    return df

def AmenitiesDataProducer(df):
    df = df.rename(columns={"resoFacts/hasAttachedProperty": "hasAttachedProperty", "resoFacts/hasView": "hasView", "resoFacts/hasGarage":"hasGarage"})
    bool_cols = ["hasGarage", "hasAttachedProperty", "hasView"]
    df[bool_cols] = df[bool_cols].astype("boolean")
    return df

def MiscDataProducer(df):
    df.drop("city", axis=1, inplace=True) # we will replace city with a different city column
    df = df.rename(columns={"resoFacts/hasHomeWarranty": "hasHomeWarranty", "address/city": "city"})
    df.loc[df["city"].str.contains("bear valley", case=False), "city"] = "Tehachapi" # Apparently the city "Bear Valley" is another word for the city "Tehachiapi"
    df["city"] = df["city"].str.lower()
    df['city'] = df['city'].replace('', pd.NA)
    df["city"] = df["city"].astype("category")
    df["homeType"] = df['homeType'].replace(pd.NA, "HOME_TYPE_UNKNOWN")
    df["yearBuilt"] = df["yearBuilt"].astype("Int64")
    df["hasHomeWarranty"] = df["hasHomeWarranty"].astype("boolean")
    df["homeType"] = df["homeType"].astype("category")
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
  """
  Calculate the haversine distance between two coordinate points
  """
  R = 6371  # Earth radius in kilometers

  lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
  dlat = lat2 - lat1
  dlon = lon2 - lon1

  a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
  c = 2 * atan2(sqrt(a), sqrt(1-a))
  return R * c * 1000  # Convert to meters

def find_nearest_city(target_city: dict, city_list: List[dict]) -> dict:
    """
    Takes a target city information as a dict containing coordinates and a list of dicts of cities and their coordinates and returns the nearest city info as dict
    """
    target_lat = target_city["latitude"]
    target_lon = target_city["longitude"]

    nearest_city = None
    shortest_distance = float('inf')

    for city in city_list:
        city_lat = city["latitude"]
        city_lon = city["longitude"]
        distance = haversine_distance(target_lat, target_lon, city_lat, city_lon)

        if distance < shortest_distance:
            shortest_distance = distance
            nearest_city = city

    return nearest_city, shortest_distance

def impute_climate_and_elevation(df: pd.DataFrame):
    missing_climate = df[df["mean_daily_high_temperature"].isna() |
                     df["mean_daily_low_temperature"].isna() |
                     df["rainfall"].isna()]

    has_climate = df.drop(missing_climate.index)
    has_climate_list = has_climate.to_dict("records")

    for index, missing_climate_row in missing_climate.iterrows():
        nearest_city_dict, distance = find_nearest_city(missing_climate_row.to_dict(), has_climate_list)
        missing_climate.loc[index, "mean_daily_high_temperature"] = nearest_city_dict["mean_daily_high_temperature"]
        missing_climate.loc[index, "mean_daily_low_temperature"] = nearest_city_dict["mean_daily_low_temperature"]
        missing_climate.loc[index, "rainfall"] = nearest_city_dict["rainfall"]
        missing_climate.loc[index, "elevation"] = nearest_city_dict["elevation"]
        # print(f"distance from {missing_climate_row['city']} to {nearest_city_dict['city']} is {distance/1000:.2f} kilometers away")

    return pd.concat([missing_climate, has_climate]).sort_index()

data = HOADataProducer(data)
data = LayoutDataProducer(data)
data = AmenitiesDataProducer(data)
data = MiscDataProducer(data)

data['latitude'] = data['latitude'].fillna(
    data['nearbyHomes/0/latitude']).fillna(
    data['nearbyHomes/1/latitude']).fillna(
    data['nearbyHomes/3/latitude'])

data['longitude'] = data['longitude'].fillna(
    data['nearbyHomes/0/longitude']).fillna(
    data['nearbyHomes/1/longitude']).fillna(
    data['nearbyHomes/3/longitude'])

data.dropna(subset=["latitude", "longitude"], inplace=True)

climate_df = pd.read_csv("../climate_population_density.csv")


imputed_climate_df = impute_climate_and_elevation(climate_df)

imputed_climate_df.drop(["latitude", "longitude"], axis=1, inplace=True)

feature_engineered_data = pd.merge(data, imputed_climate_df, on='city', how='left')

nearest_features_distance_df = pd.read_csv("../nearest_features_distance.csv")

feature_engineered_data = pd.merge(feature_engineered_data, nearest_features_distance_df, left_index=True, right_index=True, how='inner')

columns_to_keep = ['price', "lotSize", "livingArea", "hasGarage", "hasAttachedProperty", "hasView", "hasHOA", "AnnualHOAFee", "fullBathrooms", "halfBathrooms", 'bedrooms', 'city', 'homeType', 'propertyTaxRate', 'yearBuilt', 'hasHomeWarranty', 'latitude', 'longitude', 'elevation', 'mean_daily_high_temperature', 'mean_daily_low_temperature', 'rainfall', 'population_density', 'park', 'motorway', 'supermarket', 'Costco']
df_keep = feature_engineered_data.loc[:, columns_to_keep]

# Only train on homes that are actually sold for the price
df_keep = df_keep.loc[feature_engineered_data["homeStatus"] == "RECENTLY_SOLD"]

# Filter out properties that don't have buildings on them
df_keep = df_keep.loc[df_keep["homeType"] != "LOT"]

# Filter out weird yearBuilt datapoints
df_keep = df_keep.loc[df_keep["yearBuilt"] > 0]

# Filter out where property taxes are not given
df_keep = df_keep.loc[(df_keep["propertyTaxRate"] > 0) | (df_keep["propertyTaxRate"] == np.nan)]

# Calculate Q1 (25th percentile) and Q3 (75th percentile) for each column
Q1_living = df_keep['livingArea'].quantile(0.25)
Q3_living = df_keep['livingArea'].quantile(0.75)
IQR_living = Q3_living - Q1_living

Q1_lot = df_keep['lotSize'].quantile(0.25)
Q3_lot = df_keep['lotSize'].quantile(0.75)
IQR_lot = Q3_lot - Q1_lot

# Define the lower and upper bounds for outliers
lower_bound_living = Q1_living - 1.5 * IQR_living
upper_bound_living = Q3_living + 1.5 * IQR_living

lower_bound_lot = Q1_lot - 1.5 * IQR_lot
upper_bound_lot = Q3_lot + 1.5 * IQR_lot

# Filter the dataframe based on the IQR bounds for both columns
df_keep = df_keep.loc[
    (df_keep['livingArea'] >= lower_bound_living) & (df_keep['livingArea'] <= upper_bound_living) &
    (df_keep['lotSize'] >= lower_bound_lot) & (df_keep['lotSize'] <= upper_bound_lot)
]

# Manually removing livingArea outlier using domain knowlege (a.k.a no one can live in a <1sqft space)
df_keep = df_keep.loc[df_keep['livingArea'] > 50]

df_keep.to_csv("housing.csv", index=False)
