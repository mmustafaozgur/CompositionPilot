import pandas as pd
import numpy as np
import os
import random # generate_random_composition için
import joblib # Model, Scaler ve özellik listesini kaydetmek/yüklemek için
from catboost import CatBoostRegressor # Sadece tip ipucu veya bazı özel durumlar için gerekebilir

#-------------------------------------------------------------------------------
# Blok 2'den Fonksiyon (Jupyter Notebook'tan alındı - generate_random_composition)
#-------------------------------------------------------------------------------
def generate_random_composition(elements, min_frac=0.01, n_elements_limit=None):
    """
    Belirtilen elementler listesinden rastgele bir kompozisyon oluşturur.
    Her elementin minimum fraksiyonu min_frac ile sınırlıdır ve toplamları 1.0 olur.
    n_elements_limit: Kompozisyonda bulunacak maksimum element sayısı. None ise sınırsız.
    """
    if not elements:
        return {}

    if n_elements_limit is None or n_elements_limit > len(elements):
        n_elements_limit = len(elements)
    
    if n_elements_limit <= 0: 
        if elements : 
             num_selected_elements = random.randint(1, max(1,len(elements)))
        else: 
            return {}
    else: 
        num_selected_elements = random.randint(1, n_elements_limit)

    selected_elements = random.sample(elements, num_selected_elements)
    
    if not selected_elements:
        return {}

    fractions = np.random.dirichlet(np.ones(len(selected_elements)), size=1)[0]
    
    composition = {}
    adjusted_fractions = [max(frac, min_frac) for frac in fractions]
    
    total_adjusted = sum(adjusted_fractions)
    if total_adjusted == 0: 
        if len(selected_elements) > 0:
            equal_frac = 1.0 / len(selected_elements)
            for elem in selected_elements:
                composition[elem] = max(equal_frac, min_frac)
            current_sum_final = sum(composition.values())
            if current_sum_final > 0 :
                for elem in composition:
                    composition[elem] /= current_sum_final
        return composition

    final_fractions = [f / total_adjusted for f in adjusted_fractions]

    for i, elem in enumerate(selected_elements):
        composition[elem] = final_fractions[i]
            
    current_sum_final = sum(composition.values())
    if current_sum_final > 0 and abs(current_sum_final - 1.0) > 1e-9 : 
        for elem in composition:
            composition[elem] /= current_sum_final
                
    return composition

#-------------------------------------------------------------------------------
# Blok 3'ten Fonksiyon (Jupyter Notebook'tan alındı - find_min_delta_e_random)
#-------------------------------------------------------------------------------
def find_min_delta_e_random(model, elements_to_vary, all_feature_cols, scaler,
                            n_iter=10000, min_frac=0.01, fixed_features=None):
    """
    Rastgele kompozisyonlar üreterek ve CatBoost modeliyle tahmin yaparak en düşük Delta E'yi bulur.
    fixed_features: {'feature_name': value} veya {'feature_name': "dynamic_from_comp"} 
                    şeklinde sabitlenecek/dinamik ayarlanacak ek özellikler.
    """
    min_delta_e = float('inf')
    best_composition_details = None
    
    if not elements_to_vary:
        print("Uyarı: `elements_to_vary` listesi boş. Rastgele arama yapılamıyor.")
        return min_delta_e, best_composition_details

    print(f"{n_iter} iterasyon boyunca rastgele kompozisyonlar deneniyor...")
    for i in range(n_iter):
        current_composition = generate_random_composition(elements_to_vary, min_frac=min_frac)
        if not current_composition:
            continue

        input_features = {col: 0.0 for col in all_feature_cols}
        num_elements_in_comp = 0
        for elem, frac in current_composition.items():
            if elem in input_features:
                input_features[elem] = frac
                if frac > 1e-6:
                    num_elements_in_comp +=1
        
        if fixed_features:
            for feat, val in fixed_features.items():
                if feat in input_features:
                    if val == "dynamic_from_comp" and feat == 'comp_ntypes':
                         input_features[feat] = num_elements_in_comp
                    else:
                        input_features[feat] = val
        
        if 'comp_ntypes' in input_features and \
           (not fixed_features or 'comp_ntypes' not in fixed_features or \
            (fixed_features and fixed_features.get('comp_ntypes') != "dynamic_from_comp")):
            input_features['comp_ntypes'] = num_elements_in_comp

        df_input = pd.DataFrame([input_features], columns=all_feature_cols)
        df_input_scaled = scaler.transform(df_input)
        
        try:
            predicted_delta_e_val = model.predict(df_input_scaled)
            current_delta_e = predicted_delta_e_val[0] if isinstance(predicted_delta_e_val, np.ndarray) and predicted_delta_e_val.ndim > 0 else predicted_delta_e_val
        except Exception as e:
            print(f"Tahmin sırasında hata: {e}, kompozisyon: {current_composition}")
            continue 

        if current_delta_e < min_delta_e:
            min_delta_e = current_delta_e
            best_composition_details = input_features.copy()

        if (i + 1) % 10 == 0 or (i + 1) == n_iter:
            print(f"Iterasyon {i+1}/{n_iter}... En düşük ΔE: {min_delta_e:.4f}")
    
    if best_composition_details:
        print(f"\nEn Düşük ΔE (CatBoost - Rastgele Arama): {min_delta_e:.4f}")
        display_comp = {
            k: v for k, v in best_composition_details.items() 
            if (k in elements_to_vary and v > 1e-6) or \
               (k not in elements_to_vary and v != 0.0 and (fixed_features and k in fixed_features)) or \
               (k == 'comp_ntypes' and 'comp_ntypes' in best_composition_details)
        }
        print(f"Bulunan Kompozisyon (ve ilgili özellikler): {display_comp}")
    else:
        print("CatBoost - Rastgele arama sonucu uygun kompozisyon bulunamadı.")
            
    return min_delta_e, best_composition_details

#-------------------------------------------------------------------------------
# Ana Çalıştırma Bloğu
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    MODEL_FILENAME = 'catboost_delta_e_model.joblib' 
    SCALER_FILENAME = 'catboost_scaler.joblib'
    FEATURES_FILENAME = 'catboost_model_features.joblib'

    trained_cb_model = None
    trained_cb_scaler = None
    cb_model_features = None

    try:
        print(f"CatBoost Modeli '{MODEL_FILENAME}' dosyasından yükleniyor...")
        trained_cb_model = joblib.load(MODEL_FILENAME) # DÜZELTİLDİ: joblib.load()
        print("CatBoost Modeli yüklendi.")
        
        print(f"Ölçekleyici '{SCALER_FILENAME}' dosyasından yükleniyor...")
        trained_cb_scaler = joblib.load(SCALER_FILENAME)
        print("Ölçekleyici yüklendi.")
        
        print(f"Model özellikleri '{FEATURES_FILENAME}' dosyasından yükleniyor...")
        cb_model_features = joblib.load(FEATURES_FILENAME)
        print("Model özellikleri yüklendi.")
        
    except FileNotFoundError as e:
        print(f"HATA: Gerekli dosya bulunamadı: {e}. Lütfen modelin, ölçekleyicinin ve özellik dosyasının mevcut olduğundan emin olun.")
        exit()
    except Exception as e:
        print(f"Dosyalar yüklenirken bir hata oluştu: {e}")
        exit()

    if trained_cb_model and trained_cb_scaler and cb_model_features:
        print("\n--- Rastgele Kompozisyonlarla En Düşük Delta E Arama (CatBoost ile) Başlatılıyor ---")
        
        possible_elements_cb = ['Fe', 'Ni', 'Al', 'Cr']
        elements_for_search_cb = [elem for elem in possible_elements_cb if elem in cb_model_features]

        if elements_for_search_cb:
            print(f"Arama için kullanılacak elementler (modelde olanlar ve listeden seçilenler): {elements_for_search_cb}")
            
            fixed_s_features_cb = {}
            if 'comp_ntypes' in cb_model_features:
                fixed_s_features_cb['comp_ntypes'] = "dynamic_from_comp" 
            print(f"Rastgele arama için sabit/dinamik özellikler: {fixed_s_features_cb}")

            n_iterations_cb = 1000 
            min_element_fraction_cb = 0.02

            min_delta_e_cb, best_comp_cb = find_min_delta_e_random(
                model=trained_cb_model,
                elements_to_vary=elements_for_search_cb,
                all_feature_cols=cb_model_features,
                scaler=trained_cb_scaler,
                n_iter=n_iterations_cb,
                min_frac=min_element_fraction_cb,
                fixed_features=fixed_s_features_cb
            )
        else:
            print("Model özelliklerinde veya tanımlı listede arama için uygun element bulunamadı.")
    else:
        print("\nModel, ölçekleyici veya özellikler yüklenemediği için arama başlatılamıyor.")