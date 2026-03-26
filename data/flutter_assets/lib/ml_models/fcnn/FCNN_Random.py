# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt  # Not needed for API functionality
# import seaborn as sns  # Not needed for API functionality
import os
import random # generate_random_composition için
import joblib # Scaler ve özellik listesini kaydetmek/yüklemek için

import tensorflow as tf
from tensorflow.keras.models import load_model # Model yükleme için
# Huber gibi özel kayıp fonksiyonları compile=False ile yüklenirken sorun çıkarmaz
# ancak gerekirse custom_objects={'Huber': Huber} şeklinde eklenebilir.
# Modeliniz compile edilmiş kaydedildiği için compile=False gerekmeyebilir.

#-------------------------------------------------------------------------------
# Blok 5'ten Fonksiyon (Jupyter Notebook'tan alındı - generate_random_composition)
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
    
    # Kompozisyonda yer alacak element sayısını rastgele belirle (en az 1)
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
# Blok 6'dan Fonksiyon (Jupyter Notebook'tan alındı - find_min_delta_e_random_mlp)
#-------------------------------------------------------------------------------
def find_min_delta_e_random_mlp(model, elements_to_vary, all_feature_cols, scaler,
                                n_iter=10000, min_frac=0.01, fixed_features=None):
    """
    Rastgele kompozisyonlar üreterek ve MLP modeliyle tahmin yaparak en düşük Delta E'yi bulur.
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
            predicted_delta_e = model.predict(df_input_scaled, verbose=0).flatten() # MLP için .flatten() ve TensorFlow uyarılarını gizlemek için verbose=0
            current_delta_e = predicted_delta_e[0] 
        except Exception as e:
            print(f"Tahmin sırasında hata: {e}, kompozisyon: {current_composition}")
            continue 

        if current_delta_e < min_delta_e:
            min_delta_e = current_delta_e
            best_composition_details = input_features.copy()

        # Her 10 iterasyonda bir veya son iterasyonda ilerleme durumunu yazdır
        if (i + 1) % 10 == 0 or (i + 1) == n_iter: 
            print(f"Iterasyon {i+1}/{n_iter}... En düşük Delta E: {min_delta_e:.4f}")
    
    if best_composition_details:
        print(f"\nEn Düşük Delta E (MLP - Rastgele Arama): {min_delta_e:.4f}")
        display_comp = {
            k: v for k, v in best_composition_details.items() 
            if (k in elements_to_vary and v > 1e-6) or \
               (k not in elements_to_vary and v != 0.0 and (fixed_features and k in fixed_features)) or \
               (k == 'comp_ntypes' and 'comp_ntypes' in best_composition_details)
        }
        print(f"Bulunan Kompozisyon (ve ilgili özellikler): {display_comp}")
    else:
        print("MLP - Rastgele arama sonucu uygun kompozisyon bulunamadı.")
            
    return min_delta_e, best_composition_details

#-------------------------------------------------------------------------------
# Ana Çalıştırma Bloğu
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    MODEL_FILENAME = 'mlp_delta_e_model_lasso.h5'
    SCALER_FILENAME = 'mlp_scaler_lasso.joblib'
    FEATURES_FILENAME = 'mlp_model_features_lasso.joblib'

    trained_mlp_model = None
    trained_final_scaler = None
    selected_model_features = None

    try:
        print(f"MLP Modeli '{MODEL_FILENAME}' dosyasından yükleniyor...")
        trained_mlp_model = load_model(MODEL_FILENAME) # Keras modeli yükleme
        print("MLP Modeli yüklendi.")
        
        print(f"Ölçekleyici '{SCALER_FILENAME}' dosyasından yükleniyor...")
        trained_final_scaler = joblib.load(SCALER_FILENAME)
        print("Ölçekleyici yüklendi.")
        
        print(f"Model özellikleri '{FEATURES_FILENAME}' dosyasından yükleniyor...")
        selected_model_features = joblib.load(FEATURES_FILENAME)
        print("Model özellikleri yüklendi.")
        
    except FileNotFoundError as e:
        print(f"HATA: Gerekli dosya bulunamadı: {e}. Lütfen modelin, ölçekleyicinin ve özellik dosyasının mevcut olduğundan emin olun.")
        exit()
    except Exception as e:
        print(f"Dosyalar yüklenirken bir hata oluştu: {e}")
        exit()

    if trained_mlp_model and trained_final_scaler and selected_model_features:
        print("\n--- Rastgele Kompozisyonlarla En Düşük Delta E Arama (MLP ile) Başlatılıyor ---")
        
        possible_elements = ['Fe', 'Ni', 'Al', 'Cr', 'Co']
        # Modelin bildiği ve arama yapılacak elementler listesinde olanları filtrele
        elements_for_search_mlp = [elem for elem in possible_elements if elem in selected_model_features]

        if elements_for_search_mlp:
            print(f"Arama için kullanılacak elementler (modelde olanlar ve listeden seçilenler): {elements_for_search_mlp}")
            
            fixed_s_features_mlp = {}
            if 'comp_ntypes' in selected_model_features:
                fixed_s_features_mlp['comp_ntypes'] = "dynamic_from_comp" 
            # fixed_s_features_mlp['temperature'] = 273 # Örnek sabit özellik
            print(f"Rastgele arama için sabit/dinamik özellikler: {fixed_s_features_mlp}")

            n_iterations_mlp = 1000 
            min_element_fraction_mlp = 0.02

            min_delta_e_mlp, best_comp_mlp = find_min_delta_e_random_mlp(
                model=trained_mlp_model,
                elements_to_vary=elements_for_search_mlp,
                all_feature_cols=selected_model_features,
                scaler=trained_final_scaler,
                n_iter=n_iterations_mlp,
                min_frac=min_element_fraction_mlp,
                fixed_features=fixed_s_features_mlp
            )
        else:
            print("Model özelliklerinde veya tanımlı listede arama için uygun element bulunamadı.")
    else:
        print("\nModel, ölçekleyici veya özellikler yüklenemediği için arama başlatılamıyor.")