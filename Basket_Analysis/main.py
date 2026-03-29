import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules


# KROK 1: WCZYTANIE I CZYSZCZENIE DANYCH
print("Wczytywanie i czyszczenie danych...")
data = pd.read_csv("data.csv", encoding="latin1")

# Naprawa opisów i usuwanie braków
data['Description'] = data['Description'].fillna("BRAK OPISU").astype(str).str.strip()
data.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
data['InvoiceNo'] = data["InvoiceNo"].astype("str")

# Usuwanie anulowanych zamówień i kodów operacyjnych
data = data[~data["InvoiceNo"].str.contains("C")]
kody_do_usuniecia = ['POST', 'D', 'M', 'CRUK', 'DOT', 'BANK CHARGES', 'AMAZON FEE']
data = data[~data['StockCode'].isin(kody_do_usuniecia)]

# Zawężenie do głównego rynku (UK) dla lepszej dokładności
data = data[data['Country'] == 'United Kingdom']


# KROK 2: BUDOWA SŁOWNIKA I MACIERZY KOSZYKA
print("Budowanie macierzy koszyka zakupowego...")

# Słownik tłumaczący: StockCode -> Description
item_map = data.sort_values('InvoiceDate').drop_duplicates(subset=['StockCode'], keep='last')
item_map = item_map.set_index('StockCode')['Description'].to_dict()

# Macierz zer i jedynek
basket = data.groupby(["InvoiceNo", "StockCode"])["Quantity"].sum().unstack().fillna(0)

def encode_units(x):
    return 1 if x >= 1 else 0

basket_sets = basket.map(encode_units)


#TRENING MODELU 

print("Trenowanie algorytmu FP-Growth i szukanie powiązań...")

# Szukanie częstych zestawów (min. 2% koszyków)
frequent_itemsets = fpgrowth(basket_sets, min_support=0.02, use_colnames=True)

# Generowanie reguł i filtrowanie po mnożniku sprzedaży (Lift > 1)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])


#tłumaczenie wyniku
def map_codes_to_names(frozen_set_of_codes, dictionary):
    return " + ".join([str(dictionary.get(code, "Nieznany Produkt")) for code in frozen_set_of_codes])

rules['Produkt_w_koszyku'] = rules['antecedents'].apply(lambda x: map_codes_to_names(x, item_map))
rules['Rekomendacja'] = rules['consequents'].apply(lambda x: map_codes_to_names(x, item_map))

# Wyświetlanie gotowego raportu
kolumny_biznesowe = ['Produkt_w_koszyku', 'Rekomendacja', 'support', 'confidence', 'lift']

print("\n--- 🎯 TOP 10 REKOMENDACJI CROSS-SELL DLA BIZNESU ---")
print(rules[kolumny_biznesowe].head(10).to_string(index=False))