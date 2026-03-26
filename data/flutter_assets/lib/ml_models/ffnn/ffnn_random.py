import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

def generate_random_composition(elements, min_frac=0.05):
    """
    elements: ['Fe', 'Ni', 'Al', ...]
    min_frac: her elementin alması gereken minimum oran
    """
    n = len(elements)
    if n * min_frac > 1:
        raise ValueError("Minimum pay, element sayısı için çok yüksek!")
    fixed_total = n * min_frac
    remainder = 1 - fixed_total
    props = np.random.dirichlet(np.ones(n))
    return {e: min_frac + p * remainder for e, p in zip(elements, props)}

def find_min_delta_e_with_random_nn(
    elements,
    model_path="nn_model.h5",
    columns_csv="columns_data.csv",
    data_csv="yeni_xgboost_veri.csv",
    n_iter=1000,
    min_frac=0.05
):
    """
    elements:       kompozisyonda kullanmak istediğiniz element listesi
    model_path:     kaydettiğiniz Keras .h5 modeli
    columns_csv:    sadece feature sütun adlarını içeren CSV (delta_e hariç)
    data_csv:       model eğitiminde kullandığınız tam veri CSV'si
    n_iter:         kaç rastgele örnek denensin
    min_frac:       her elementin minimum oranı
    """
    # 1) feature sütun adlarını oku
    df_cols = pd.read_csv(columns_csv, nrows=0)
    feature_cols = [c for c in df_cols.columns]

    # 2) scaler'ı tüm eğitim verisi üzerinde fit et
    df_full = pd.read_csv(data_csv)
    X_full = df_full[feature_cols].values
    scaler = StandardScaler().fit(X_full)

    # 3) modeli yükle (compile=False ile metriği es geç)
    model = load_model(model_path, compile=False)

    best_delta = float("inf")
    best_comp  = None

    # indeks haritası hız için
    idx_map = {f: i for i, f in enumerate(feature_cols)}

    for i in range(1, n_iter + 1):
        # 4) rastgele kompozisyon üret
        comp = generate_random_composition(elements, min_frac)

        # 5) feature vektörünü oluştur
        x = np.zeros((1, len(feature_cols)), dtype=float)
        for elem, frac in comp.items():
            if elem not in idx_map:
                raise KeyError(f"'{elem}' feature_cols içinde yok!")
            x[0, idx_map[elem]] = frac

        # 6) ölçekle ve tahmin et
        x_scaled = scaler.transform(x)
        pred = float(model.predict(x_scaled)[0, 0])

        # 7) en iyiyi güncelle
        if pred < best_delta:
            best_delta = pred
            best_comp  = comp.copy()

        print(f"Iterasyon {i}/{n_iter}: ΔE = {pred:.5f}, kompozisyon = {comp}")

    return best_delta, best_comp

# --------------- KULLANIM ÖRNEĞİ ---------------

if __name__ == "__main__":
    elements = ["Fe", "Ni", "Al"]  # kullanmak istediğiniz elementler
    best_delta_e, best_composition = find_min_delta_e_with_random_nn(
        elements=elements,
        model_path="nn_model.h5",
        columns_csv="columns_data.csv",
        data_csv="yeni_xgboost_veri.csv",
        n_iter=100,    # istediğiniz iterasyon sayısı
        min_frac=0.05  # her element için minimum oran
    )
    print("\n=== Sonuç ===")
    print(f"En düşük tahmini ΔE: {best_delta_e:.5f} eV")
    print("Kompozisyon:", best_composition)
    print("Toplam oran:", sum(best_composition.values()))
