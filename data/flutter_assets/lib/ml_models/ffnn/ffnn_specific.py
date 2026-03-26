import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

def predict_delta_e_nn_from_dict(
    comp_dict: dict,
    model_path: str = "nn_model.h5",
    data_csv:   str = "yeni_xgboost_veri.csv",
    cols_csv:   str = "columns_data.csv"
) -> float:
    """
    comp_dict:   {'Fe':0.5, 'Ni':0.3, 'Al':0.2, ...} (toplam 1.0)
    model_path:  Kayıtlı Keras model dosyanız
    data_csv:    nn modelini eğitirken kullandığınız tam veri CSV’si
    cols_csv:    Sadece feature sütun adlarını (delta_e hariç) tutan CSV
    """
    # 1) Feature sütun adlarını al:
    df_cols = pd.read_csv(cols_csv, nrows=0)
    feature_cols = [c for c in df_cols.columns if c != "delta_e"]

    # 2) comp_dict toplam kontrolü:
    total = sum(comp_dict.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Oranlar toplamı 1 değil: {total:.6f}")

    # 3) Girdi vektörünü oluştur:
    x = np.zeros((1, len(feature_cols)), dtype=float)
    for i, feat in enumerate(feature_cols):
        if feat in comp_dict:
            x[0, i] = comp_dict[feat]

    # 4) Scaler’ı yeniden hesapla:
    df_full = pd.read_csv(data_csv)
    X_full  = df_full[feature_cols].values
    scaler  = StandardScaler().fit(X_full)
    x_scaled = scaler.transform(x)

    # 5) Modeli yükle (compile=False ile metric hatasından kaçın):
    model = load_model(model_path, compile=False)

    # 6) Tahmini al:
    delta_e_pred = model.predict(x_scaled)[0, 0]
    return float(delta_e_pred)


if __name__ == "__main__":
    # Örnek kullanım
    comp = {"Fe": 0.333333333333333, "Ac": 0.333333333333333, "Ag": 0.333333333333333}
    pred = predict_delta_e_nn_from_dict(
        comp_dict=comp,
        model_path="nn_model.h5",
        data_csv="yeni_xgboost_veri.csv",
        cols_csv="columns_data.csv"
    )
    print(f"Tahmin edilen formasyon enerjisi (ΔE): {pred:.5f} eV")
