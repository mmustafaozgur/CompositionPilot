#-------------------------------------------------------------------------------
# random_generate_delta_e.py
#-------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import joblib
import os
import random

#-------------------------------------------------------------------------------
# Blok 2'den Fonksiyonlar (Jupyter Notebook'tan alındı)
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
    # n_elements_limit 0 veya negatif olamayacağı için, eğer 1 ise randint(1,1) olacak.
    if n_elements_limit <= 0: # Eğer limit anlamsızsa veya boşsa
        if elements : # Element varsa en az 1 element seç
             num_selected_elements = random.randint(1, max(1,len(elements)))
        else: # Element yoksa boş dön
            return {}
    else: # Limit anlamlıysa
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


def find_min_delta_e_random(model, elements_to_vary, all_feature_cols, scaler,
                            n_iter=10000, min_frac=0.01, fixed_features=None):
    """
    Rastgele kompozisyonlar üreterek ve modelle tahmin yaparak en düşük Delta E'yi bulur.
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
                if frac > 1e-6: # Çok küçük fraksiyonları sayma
                    num_elements_in_comp +=1
        
        if fixed_features:
            for feat, val in fixed_features.items():
                if feat in input_features:
                    if val == "dynamic_from_comp" and feat == 'comp_ntypes':
                         input_features[feat] = num_elements_in_comp
                    else:
                        input_features[feat] = val
        
        # Eğer fixed_features'ta comp_ntypes belirtilmemişse veya dynamic değilse,
        # ve comp_ntypes model özelliklerindeyse, yine de num_elements_in_comp ile ayarla.
        # Bu kısım, comp_ntypes'ın her zaman güncel olmasını sağlamaya yardımcı olur.
        if 'comp_ntypes' in input_features and \
           (not fixed_features or 'comp_ntypes' not in fixed_features or \
            (fixed_features and fixed_features.get('comp_ntypes') != "dynamic_from_comp")):
            input_features['comp_ntypes'] = num_elements_in_comp


        df_input = pd.DataFrame([input_features], columns=all_feature_cols)
        df_input_scaled = scaler.transform(df_input)
        
        try:
            predicted_delta_e = model.predict(df_input_scaled)
            current_delta_e = predicted_delta_e[0] 
        except Exception as e:
            print(f"Tahmin sırasında hata: {e}, kompozisyon: {current_composition}")
            continue 

        if current_delta_e < min_delta_e:
            min_delta_e = current_delta_e
            best_composition_details = input_features.copy() # Store a copy

        if (i + 1) % (max(1, n_iter // 10)) == 0: # n_iter 0 ise veya 10'dan küçükse max(1,0) -> 1 olur
             print(f"Iterasyon {i+1}/{n_iter}... En düşük ΔE: {min_delta_e:.4f}")
    
    if best_composition_details:
        print(f"\nEn Düşük ΔE (Rastgele Arama): {min_delta_e:.4f}")
        # Sadece element olan ve fraksiyonu > 0 olanları veya comp_ntypes gibi sabit/dinamik özellikleri göster
        display_comp = {
            k: v for k, v in best_composition_details.items() 
            if (k in elements_to_vary and v > 1e-6) or \
               (k not in elements_to_vary and v != 0.0 and (fixed_features and k in fixed_features)) or \
               (k == 'comp_ntypes' and 'comp_ntypes' in best_composition_details) # comp_ntypes'ı her zaman göster
        }
        # Eğer 'comp_ntypes' için özel bir koşul varsa veya her zaman göstermek istiyorsan, yukarıdaki koşulu düzenle
        # Örneğin, sadece elementleri ve comp_ntypes'ı göstermek için:
        # display_comp = {k: v for k, v in best_composition_details.items() if (k in elements_to_vary and v > 1e-6) or k == 'comp_ntypes'}

        print(f"Bulunan Kompozisyon (ve ilgili özellikler): {display_comp}")
    else:
        print("Rastgele arama sonucu uygun kompozisyon bulunamadı.")
            
    return min_delta_e, best_composition_details

#-------------------------------------------------------------------------------
# Ana Çalıştırma Bloğu
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    MODEL_FILENAME = 'lightgbm_delta_e_model.txt'
    SCALER_FILENAME = 'lightgbm_scaler.joblib'
    FEATURES_FILENAME = 'lightgbm_model_features.joblib'

    trained_model = None
    trained_scaler = None
    model_feature_cols = None

    try:
        print(f"Model '{MODEL_FILENAME}' dosyasından yükleniyor...")
        trained_model = joblib.load(MODEL_FILENAME)
        print("Model yüklendi.")
        
        print(f"Ölçekleyici '{SCALER_FILENAME}' dosyasından yükleniyor...")
        trained_scaler = joblib.load(SCALER_FILENAME)
        print("Ölçekleyici yüklendi.")
        
        print(f"Model özellikleri '{FEATURES_FILENAME}' dosyasından yükleniyor...")
        model_feature_cols = joblib.load(FEATURES_FILENAME)
        print("Model özellikleri yüklendi.")
        
    except FileNotFoundError as e:
        print(f"HATA: Gerekli dosya bulunamadı: {e}. Lütfen modelin, ölçekleyicinin ve özellik dosyasının mevcut olduğundan emin olun.")
        exit()
    except Exception as e:
        print(f"Dosyalar yüklenirken bir hata oluştu: {e}")
        exit()

    if trained_model and trained_scaler and model_feature_cols:
        print("\n--- Rastgele Kompozisyonlarla En Düşük Delta E Arama Başlatılıyor ---")
        
        possible_elements = ['Fe', 'Ni', 'Al', 'Cr', 'Co', 'Si', 'C', 'Mo', 'W', 'V', 'Mn', 'Ti', 'Nb', 'Zr', 'Hf', 'Ta', 'Re', 'Ru', 'B', 'P', 'S']
        elements_for_search = [elem for elem in possible_elements if elem in model_feature_cols]

        if elements_for_search:
            print(f"Arama için kullanılacak elementler (modelde olanlar ve listeden seçilenler): {elements_for_search}")
            
            fixed_search_features = {}
            if 'comp_ntypes' in model_feature_cols:
                fixed_search_features['comp_ntypes'] = "dynamic_from_comp" 
            
            print(f"Rastgele arama için sabit/dinamik özellikler: {fixed_search_features}")

            # Parametreleri isteğe göre ayarla
            n_iterations = 1000  # Denenecek rastgele kompozisyon sayısı
            min_element_fraction = 0.02 # Her element için minimum fraksiyon

            min_delta_e_val, best_comp_details = find_min_delta_e_random(
                model=trained_model,
                elements_to_vary=elements_for_search,
                all_feature_cols=model_feature_cols,
                scaler=trained_scaler,
                n_iter=n_iterations,
                min_frac=min_element_fraction,
                fixed_features=fixed_search_features
            )
        else:
            print("Model özelliklerinde veya tanımlı listede arama için uygun element bulunamadı.")
    else:
        print("\nModel, ölçekleyici veya özellikler yüklenemediği için arama başlatılamıyor.")