import xgboost as xgb
import pandas as pd

# 1) Modeli yükleyin
model_path = "xgboost_trained_model.json"
bst = xgb.Booster()
bst.load_model(model_path)

# 2) feature_cols'u CSV'den okuyun
#    columns_data.csv dosyası yalnızca modelin kullandığı sütun adlarını içeriyor olmalı
df_cols = pd.read_csv("columns_data.csv", nrows=0)
feature_cols = df_cols.columns.tolist()

# 3) Manuel kompozisyonunuzu burada oluşturun (oranlar toplamı 1.0 olmalı)
comp_dict = {
    "Fe": 0.50,
    "Ni": 0.30,
    "Al": 0.20,
    # ... eğer başka elementler varsa buraya ekleyin
}

# 4) Toplam kontrolü (isteğe bağlı)
total = sum(comp_dict.values())
if abs(total - 1.0) > 1e-6:
    raise ValueError(f"Oranlar toplamı 1 değil: {total:.6f}")

# 5) Feature vektörünü oluşturun
row = {col: 0.0 for col in feature_cols}
for elem, frac in comp_dict.items():
    if elem not in row:
        raise KeyError(f"'{elem}' feature_cols içinde yok!")
    row[elem] = frac

# 6) DataFrame ve DMatrix'e dönüştürüp tahmin alın
df_input = pd.DataFrame([row], columns=feature_cols)
dmat = xgb.DMatrix(df_input)
delta_e_pred = bst.predict(dmat)[0]

print(f"Tahmin edilen formasyon enerjisi (ΔE): {delta_e_pred:.5f} eV")
