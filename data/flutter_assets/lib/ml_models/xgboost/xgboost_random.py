import xgboost as xgb
import pandas as pd
import numpy as np

def generate_random_composition(elements, min_frac=0.05):
    """
    elements: ['Fe', 'Ni', 'Al', ...] gibi element isimleri
    min_frac: her elementin alması gereken minimum oran
    """
    n = len(elements)
    if n * min_frac > 1:
        raise ValueError("Minimum pay, element sayısı için çok yüksek!")
    
    # Her elemente verilecek sabit pay
    fixed_total = n * min_frac
    # Kalan oran
    remainder = 1 - fixed_total
    # Dirichlet ile kalan oranı rastgele orantılara dağıtıyoruz
    random_props = np.random.dirichlet(np.ones(n))
    composition = {elem: min_frac + r * remainder for elem, r in zip(elements, random_props)}
    return composition

def find_min_delta_e_with_random(model, elements, feature_cols, n_iter=1000, min_frac=0.05):
    """
    model:      XGBoost Booster (xgb.train() ile eğitilmiş)
    elements:   ['Fe', 'Ni', 'Al'] gibi element listesi
    feature_cols: Modelin eğitiminde kullanılan DataFrame sütunları
    n_iter:     rastgele örnekleme sayısı
    min_frac:   her element için minimum fraksiyon
    """
    best_delta_e = float("inf")
    best_comp = None
    for i in range(n_iter):
        comp_dict = generate_random_composition(elements, min_frac)
        # Özellik vektörünü oluşturuyoruz:
        row_dict = {col: 0.0 for col in feature_cols}
        for elem, frac in comp_dict.items():
            if elem in row_dict:
                row_dict[elem] = frac

        df_input = pd.DataFrame([row_dict])
        dmatrix_input = xgb.DMatrix(df_input)
        
        # Tahmin yapıyoruz:
        delta_e_pred = model.predict(dmatrix_input)[0]
        if delta_e_pred < best_delta_e:
            best_delta_e = delta_e_pred
            best_comp = comp_dict
        
        best_comp = {k: float(v) for k, v in best_comp.items()}

    return best_delta_e, best_comp

# --------------- KULLANIM ÖRNEĞİ ---------------
#
# Örneğin; modelinizi eğittikten veya yükledikten sonra aşağıdaki kodu çalıştırabilirsiniz.
# 'bst' xgb.train() sonucu elde ettiğiniz Booster nesnesi olsun.
# 'X_train' model eğitimi için kullanılan DataFrame olsun.

elements = ["Fe", "Ni", "Ti"]  # Kompozisyonda kullanılacak elementler

df = pd.read_csv("columns_data.csv")
feature_cols = df.columns   # Modelin beklediği sütun isimleri

iter = 1000

bst = xgb.Booster()
bst.load_model("xgboost_trained_model.json")

best_delta_e, best_comp = find_min_delta_e_with_random(
    model=bst,
    elements=elements,
    feature_cols=feature_cols,
    n_iter=iter,
    min_frac=0.05
)

print("En düşük tahmini delta_e:", best_delta_e)
print("Kompozisyon:", best_comp)
print("Toplam:", sum(best_comp.values()))
