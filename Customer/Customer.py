import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib


# 1. ŁADOWANIE DANYCH
print("--- 1. ŁADOWANIE DANYCH ---")
df = pd.read_csv("customer_segmentation.csv")
print(f"Początkowy kształt danych: {df.shape}")


# 2. CZYSZCZENIE DANYCH (DATA CLEANING)
print("\n--- 2. CZYSZCZENIE DANYCH ---")
# Usuwanie braków danych
braki_przed = df.isna().sum().sum()
df.dropna(inplace=True)
print(f"Usunięto {braki_przed} brakujących wartości.")

# Konwersja daty
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)

# Obliczanie wieku
obecny_rok = datetime.now().year
df["Age"] = obecny_rok - df["Year_Birth"]

# Usuwanie wartości odstających (Outliers) i błędów w danych
df = df[(df["Age"] < 100) & (df["Income"] < 200000)]
df = df[(df["Marital_Status"] != "Absurd") & (df["Marital_Status"] != "YOLO")]

print(f"Kształt danych po czyszczeniu: {df.shape}")


# 3. INŻYNIERIA CECH (FEATURE ENGINEERING)
print("\n--- 3. TWORZENIE NOWYCH CECH ---")
df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

spend_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
df["Total_Spending"] = df[spend_cols].sum(axis=1)

df["Customer_Since_Days"] = (pd.Timestamp("today") - df["Dt_Customer"]).dt.days

df["AcceptedAny"] = df[["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]].sum(axis=1)
df["AcceptedAny"] = df["AcceptedAny"].apply(lambda x: 1 if x > 0 else 0)

# Grupy wiekowe
bins = [18, 30, 40, 50, 60, 70, 90]
labels = ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]
df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)

# 4. EXPLORATORY DATA ANALYSIS (EDA)
print("\n--- 4. GENEROWANIE WYKRESÓW EDA ---")

# Macierz korelacji
plt.figure(figsize=(8, 6))
corr = df[["Income", "Age", "Recency", "Total_Spending", "NumWebPurchases", "NumStorePurchases"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Macierz Korelacji")
plt.tight_layout()
plt.show()

# Średnie wydatki względem edukacji
plt.figure(figsize=(8, 5))
sns.barplot(x="Education", y="Total_Spending", data=df, errorbar=None, palette="viridis")
plt.title("Średnie Wydatki wg Poziomu Edukacji")
plt.xticks(rotation=45)
plt.show()


# 5. PRZYGOTOWANIE POD K-MEANS (SKALOWANIE)
print("\n--- 5. SKALOWANIE I MODELOWANIE K-MEANS ---")
features = ["Age", "Income", "Total_Spending", "NumWebPurchases", "NumStorePurchases", "NumWebVisitsMonth", "Recency"]
X = df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



"""
WCSS = []
for i in range(2, 10):
    kmeans_temp = KMeans(n_clusters=i, random_state=42)
    kmeans_temp.fit(X_scaled)
    WCSS.append(kmeans_temp.inertia_)
plt.figure(figsize=(8,5))
plt.plot(range(2, 10), WCSS, marker="o")
plt.title("Metoda Łokcia (Elbow Method) dla optymalnego K")
plt.xlabel("Liczba Klastrów (K)")
plt.show()
"""

kmeans = KMeans(n_clusters=6, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("Liczba klientów przypisanych do poszczególnych segmentów:")
print(df["Cluster"].value_counts().sort_index())

# 6. WIZUALIZACJA KLASTRÓW ZA POMOCĄ PCA
print("\n--- 6. REDUKCJA WYMIARÓW (PCA) ---")
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)
df["PCA1"] = pca_data[:, 0]
df["PCA2"] = pca_data[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=df, palette="Set1", s=60, alpha=0.8)
plt.title("Segmentacja Klientów - Wizualizacja 2D (PCA)")
plt.show()


# 7. ZAPISYWANIE MODELU DO PRODUKCJI
print("\n--- 7. ZAPISYWANIE MODELI DO PLIKÓW ---")
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Modele pomyślnie zapisane jako 'kmeans_model.pkl' oraz 'scaler.pkl'!")