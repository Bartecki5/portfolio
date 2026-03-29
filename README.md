# 📊 Data Science & Data Engineering Portfolio

Witaj w moim repozytorium! Znajdziesz tutaj zbiór moich skryptów, automatyzacji oraz projektów analitycznych napisanych w języku Python. Każdy folder to osobny mini-projekt rozwiązujący konkretny problem biznesowy lub techniczny.

## 📁 Spis projektów

| Nr | Nazwa Projektu | Krótki opis | Technologie |
|:---|:---|:---|:---|
| **01** | [CRM Data Pipeline & Quality Check](./01_CRM_Data_Cleaning) | Zautomatyzowany skrypt ETL do czyszczenia brudnej bazy CRM. Zawiera moduł raportujący błędy (Quality Check), inteligentną deduplikację na podstawie kompletności wierszy, standaryzację tekstu oraz inżynierię cech (np. wyliczanie stażu klienta). | `Pandas`, `NumPy`, `Datetime` |
| **02** | [K-Nearest Neighbors (From Scratch)](./KNN_Algorithm) | Implementacja algorytmu KNN (K-Najbliższych Sąsiadów) napisana całkowicie od zera z wykorzystaniem wektoryzacji `NumPy`. Zawiera autorski kalkulator odległości euklidesowej oraz system głosowania większościowego. Testowano na zbiorze Iris. | `Python`, `NumPy` |
| **03** | [Random Forest & Random Search](./Random_forest) | Autorski silnik Lasu Losowego i Drzew (CART/C4.5) zbudowany od podstaw. Wykorzystuje matematykę Entropii, dekorelację drzew (pierwiastek z cech) oraz własny system strojenia hiperparametrów (Random Search). Obejmuje twardy benchmarking z gotowymi modelami `scikit-learn` na medycznych danych nowotworowych. | `Python`, `NumPy`, `Scikit-Learn` |
| **04** | [Customer Segmentation & UI](./Customer) | Uczenie nienadzorowane (Unsupervised Learning). Segmentacja bazy klientów algorytmem K-Means (optymalizacja metodą łokcia) oraz wizualizacja klastrów z użyciem PCA. Zawiera interfejs webowy (Streamlit) do predykcji segmentu dla nowego klienta na żywo. | `Python`, `Scikit-Learn`, `Streamlit` |
| **05** | [Market Basket Analysis (FP-Growth)](./Basket_Analysis) | Analiza asocjacyjna danych transakcyjnych e-commerce. Wykorzystanie algorytmu FP-Growth do odkrywania reguł współkupowania produktów. Projekt obejmuje czyszczenie danych (anulowane zamówienia, kody operacyjne), budowę macierzy rzadkiej oraz interpretację metryk biznesowych (Support, Confidence, Lift) w celu generowania rekomendacji Cross-Sell. | `Python`, `MLxtend`, `Pandas` |
---
💡 *Wszystkie skrypty zostały napisane z myślą o czytelności, optymalizacji i praktycznym zastosowaniu biznesowym.*