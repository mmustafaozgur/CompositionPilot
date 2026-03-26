import pandas as pd
import numpy as np
import os
import joblib # Model, Scaler ve özellik listesini kaydetmek/yüklemek için
from catboost import CatBoostRegressor # Sadece tip ipucu veya bazı özel durumlar için gerekebilir

#-------------------------------------------------------------------------------
# Blok 4'ten Fonksiyon (Jupyter Notebook'tan alındı - predict_delta_e_specific)
#-------------------------------------------------------------------------------
def predict_delta_e_specific(composition_dict, model, all_feature_cols, scaler, fixed_external_features=None):
    """
    Verilen spesifik bir kompozisyon ve harici sabit özellikler için CatBoost modeli ile Delta E değerini tahmin eder.
    composition_dict: {'ElementSembolü': fraksiyon, ...} SADECE element fraksiyonlarını içermeli.
    fixed_external_features: {'feature_name': value} örn: {'comp_ntypes': 3, 'temperature': 300}
    """
    print(f"\nSpesifik kompozisyon için CatBoost ile tahmin (elementler): {composition_dict}")
    if fixed_external_features:
        print(f"Kullanılan sabit/harici özellikler: {fixed_external_features}")

    input_data_dict = {feature: 0.0 for feature in all_feature_cols}
    
    active_element_keys_in_comp = []
    for elem, frac in composition_dict.items():
        if elem in input_data_dict:
            input_data_dict[elem] = frac
            if frac > 1e-6:
                 active_element_keys_in_comp.append(elem)
        else:
            print(f"Uyarı: Element '{elem}' modelin özellik listesinde bulunmuyor ve atlanacak.")

    if fixed_external_features:
        for feat, val in fixed_external_features.items():
            if feat in input_data_dict:
                input_data_dict[feat] = val
            else:
                print(f"Uyarı: Sabit özellik '{feat}' modelin özellik listesinde bulunmuyor ve atlanacak.")
    
    if 'comp_ntypes' in input_data_dict:
        if fixed_external_features and 'comp_ntypes' in fixed_external_features:
            pass
        else:
            input_data_dict['comp_ntypes'] = len(active_element_keys_in_comp)
            print(f"'comp_ntypes' otomatik olarak {len(active_element_keys_in_comp)} değerine ayarlandı.")
    
    if not active_element_keys_in_comp and not (fixed_external_features and any(feat in input_data_dict for feat in fixed_external_features)):
        print("HATA: Verilen kompozisyonda model tarafından tanınan aktif element veya sabit özellik yok. Tahmin yapılamıyor.")
        return None

    input_df = pd.DataFrame([input_data_dict], columns=all_feature_cols)
    input_df_scaled = scaler.transform(input_df)
    
    try:
        predicted_delta_e_val = model.predict(input_df_scaled)
        predicted_value = predicted_delta_e_val[0] if isinstance(predicted_delta_e_val, np.ndarray) and predicted_delta_e_val.ndim > 0 else predicted_delta_e_val
        print(f"Tahmin Edilen ΔE (CatBoost): {predicted_value:.4f}")
        return predicted_value
    except Exception as e:
        print(f"Spesifik kompozisyon için CatBoost ile tahmin sırasında hata: {e}")
        return None

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
        print("\n--- Spesifik Kompozisyonlar İçin Delta E Tahmini (CatBoost ile) Başlatılıyor ---")
        
        specific_composition_1_elements_only_cb = {'Fe': 0.15, 'Cr': 0.2, 'Ni': 0.1, 'Mo': 0.1}
        fixed_ext_features_1_cb = {} 
        if 'comp_ntypes' in cb_model_features:
            active_elements_count = len([el for el in specific_composition_1_elements_only_cb if el in cb_model_features and specific_composition_1_elements_only_cb.get(el, 0) > 1e-6])
            fixed_ext_features_1_cb['comp_ntypes'] = active_elements_count

        final_specific_comp_1_cb = {
            k: v for k, v in specific_composition_1_elements_only_cb.items() if k in cb_model_features
        }
        
        if final_specific_comp_1_cb or (fixed_ext_features_1_cb and 'comp_ntypes' in fixed_ext_features_1_cb):
            predict_delta_e_specific(
                composition_dict=final_specific_comp_1_cb,
                model=trained_cb_model,
                all_feature_cols=cb_model_features,
                scaler=trained_cb_scaler,
                fixed_external_features=fixed_ext_features_1_cb
            )
        else:
             print(f"Örnek 1: '{specific_composition_1_elements_only_cb}' için modelde tanımlı element bulunamadı veya zorunlu sabit özellik eksik, tahmin atlanıyor.")

    else:
        print("\nCatBoost Modeli, ölçekleyici veya özellikler yüklenemediği için tahmin başlatılamıyor.")