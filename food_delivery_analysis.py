# ==============================
# 📦 Import Libraries
# ==============================
import pandas as pd
import numpy as np

# ==============================
# 📂 Load Dataset
# ==============================
df = pd.read_csv("delivery_time1.csv")

# ==============================
# 🔍 Basic Data Checks
# ==============================
print("Columns:\n", df.columns)
print("\nFirst 5 rows:\n", df.head())
print("\nDataset Info:")
print(df.info())

# ==============================
# ⚠️ Data Cleaning Checks
# ==============================

# Missing Values
print("\nMissing Values:\n", df.isnull().sum())

# Duplicate Rows
print("\nDuplicate Rows:", df.duplicated().sum())

# ==============================
# 📏 Feature Engineering: Distance Calculation
# ==============================

# Haversine formula to calculate distance between two coordinates
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in KM

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

# Create new column: distance_km
df["distance_km"] = haversine(
    df["Restaurant_latitude"],
    df["Restaurant_longitude"],
    df["Delivery_location_latitude"],
    df["Delivery_location_longitude"]
)

print("\nDistance Sample:\n", df["distance_km"].head())

# ==============================
# 📊 Analysis 1: Distance vs Delivery Time
# ==============================

correlation = df[["distance_km", "Time_taken(min)"]].corr()
print("\nCorrelation (Distance vs Time):\n", correlation)

# ==============================
# 📊 Analysis 2: Vehicle Impact
# ==============================

vehicle_analysis = df.groupby("Type_of_vehicle")["Time_taken(min)"].mean().sort_values()
print("\nAverage Delivery Time by Vehicle:\n", vehicle_analysis)

# ==============================
# 📊 Analysis 3: Delivery Person Ratings Impact
# ==============================

rating_analysis = df.groupby("Delivery_person_Ratings")["Time_taken(min)"].mean().sort_index()
print("\nDelivery Time by Ratings:\n", rating_analysis)

# ==============================
# 📊 Feature Engineering: Time Categories
# ==============================

df["time_category"] = pd.cut(
    df["Time_taken(min)"],
    bins=[0, 20, 30, 45, 100],
    labels=["Fast", "Moderate", "Slow", "Very Slow"]
)

# ==============================
# 📊 Analysis 4: Time Category vs Ratings
# ==============================

print(df["distance_km"].mean())

time_rating_analysis = df.groupby("time_category", observed=False)["Delivery_person_Ratings"].mean()
print("\nAverage Ratings by Time Category:\n", time_rating_analysis)

