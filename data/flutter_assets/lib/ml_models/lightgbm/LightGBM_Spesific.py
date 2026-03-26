
#-------------------------------------------------------------------------------
# specific_composition_delta_e.py
#-------------------------------------------------------------------------------


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import joblib
import os

#-------------------------------------------------------------------------------
# Blok 3'ten Fonksiyon (Jupyter Notebook'tan alındı)
#-------------------------------------------------------------------------------
def predict_delta_e_specific(composition_dict, model, all_feature_cols, scaler):
    """
    Verilen spesifik bir kompozisyon için Delta E değerini tahmin eder.
    composition_dict: {'ElementSembolü': fraksiyon, ...} ve 'comp_ntypes' gibi ek özellikleri içerebilir.
                      Bu sözlükteki elementler zaten modelin bildiği elementler olmalıdır
                      ve 'comp_ntypes' da doğru şekilde ayarlanmış olmalıdır.
    """
    print(f"\nSpesifik kompozisyon için tahmin: {composition_dict}")

    input_data_dict = {feature: 0.0 for feature in all_feature_cols}
    valid_keys_used = 0
    
    # Sadece modelin bildiği elementleri ve özellikleri al
    # comp_ntypes gibi ek özellikler de burada işlenir
    active_element_keys_in_comp = []
    for key, value in composition_dict.items():
        if key in input_data_dict:
            input_data_dict[key] = value
            valid_keys_used +=1
            if key != 'comp_ntypes' and value > 1e-6: # Eğer elementse ve fraksiyonu anlamlıysa
                 active_element_keys_in_comp.append(key)
        else:
            print(f"Uyarı: '{key}' özelliği modelin özellik listesinde bulunmuyor ve atlanacak.")

    # comp_ntypes'ın composition_dict içinde verilip verilmediğini kontrol et
    # Eğer verilmemişse ve modelde varsa, aktif element sayısına göre ayarla
    if 'comp_ntypes' in input_data_dict and 'comp_ntypes' not in composition_dict:
        input_data_dict['comp_ntypes'] = len(active_element_keys_in_comp)
        print(f"'comp_ntypes' otomatik olarak {len(active_element_keys_in_comp)} değerine ayarlandı.")
    elif 'comp_ntypes' in composition_dict and 'comp_ntypes' in input_data_dict:
        # Eğer composition_dict'te verilmişse onu kullan (zaten input_data_dict'e aktarıldı)
        pass
    elif 'comp_ntypes' in input_data_dict : # modelde var ama composition_dict'te yoksa ve yukarıdaki koşul da sağlanmadıysa
        input_data_dict['comp_ntypes'] = len(active_element_keys_in_comp)
        print(f"Uyarı: 'comp_ntypes' girdi kompozisyonunda belirtilmedi, aktif element sayısına ({len(active_element_keys_in_comp)}) göre ayarlandı.")


    if valid_keys_used == 0 or not active_element_keys_in_comp:
        print("HATA: Verilen kompozisyonda model tarafından tanınan aktif element yok. Tahmin yapılamıyor.")
        return None

    input_df = pd.DataFrame([input_data_dict], columns=all_feature_cols)
    input_df_scaled = scaler.transform(input_df)
    
    try:
        predicted_delta_e = model.predict(input_df_scaled)
        print(f"Tahmin Edilen ΔE: {predicted_delta_e[0]:.4f}")
        return predicted_delta_e[0] if predicted_delta_e is not None else None
    except Exception as e:
        print(f"Spesifik kompozisyon için tahmin sırasında hata: {e}")
        return None

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
        print("\n--- Spesifik Kompozisyonlar İçin Delta E Tahmini Başlatılıyor ---")
        
        # Örnek 1 (Notebook'taki gibi)
        specific_composition_1_elements = {'Fe': 0.6, 'Cr': 0.2, 'Ni': 0.1, 'Mo': 0.1}
        # Sadece modelin bildiği elementleri al
        comp_to_predict_1 = {k:v for k,v in specific_composition_1_elements.items() if k in model_feature_cols}
        
        # comp_ntypes'ı aktif element sayısına göre ayarla (eğer modelde varsa)
        if 'comp_ntypes' in model_feature_cols:
            element_keys_in_comp1 = [k for k in comp_to_predict_1.keys() if k != 'comp_ntypes' and comp_to_predict_1.get(k, 0) > 1e-6]
            comp_to_predict_1['comp_ntypes'] = len(element_keys_in_comp1) 

        original_element_keys_1 = [k for k in specific_composition_1_elements.keys()]
        valid_element_keys_1 = [k for k in comp_to_predict_1.keys() if k != 'comp_ntypes' and comp_to_predict_1.get(k, 0) > 1e-6]

        if valid_element_keys_1 and len(valid_element_keys_1) >= len(original_element_keys_1) * 0.5:
            predict_delta_e_specific(
                composition_dict=comp_to_predict_1,
                model=trained_model,
                all_feature_cols=model_feature_cols,
                scaler=trained_scaler
            )
        else:
            print(f"\n'{specific_composition_1_elements}' için yeterli sayıda eşleşen özellik ({len(valid_element_keys_1)}/{len(original_element_keys_1)}) bulunamadı veya hiç geçerli element yok, tahmin atlanıyor.")

        # Örnek 2 (Notebook'taki gibi)
        another_composition_elements = {'Al': 0.5, 'Ti': 0.3, 'V': 0.2}
        comp_to_predict_2 = {k:v for k,v in another_composition_elements.items() if k in model_feature_cols}

        if 'comp_ntypes' in model_feature_cols:
            element_keys_in_comp2 = [k for k in comp_to_predict_2.keys() if k != 'comp_ntypes' and comp_to_predict_2.get(k, 0) > 1e-6]
            comp_to_predict_2['comp_ntypes'] = len(element_keys_in_comp2)
        
        original_element_keys_2 = [k for k in another_composition_elements.keys()]
        valid_element_keys_2 = [k for k in comp_to_predict_2.keys() if k != 'comp_ntypes' and comp_to_predict_2.get(k, 0) > 1e-6]

        if valid_element_keys_2 and len(valid_element_keys_2) >= len(original_element_keys_2) * 0.5 :
             predict_delta_e_specific(
                composition_dict=comp_to_predict_2,
                model=trained_model,
                all_feature_cols=model_feature_cols,
                scaler=trained_scaler
            )
        else:
            print(f"\n'{another_composition_elements}' için yeterli sayıda eşleşen özellik ({len(valid_element_keys_2)}/{len(original_element_keys_2)}) bulunamadı veya hiç geçerli element yok, tahmin atlanıyor.")
        
        # Arayüzden gelen veya manuel olarak girilecek bir kompozisyon için örnek:
        # Diyelim ki arayüzden şu elementler ve oranları geldi:
        # custom_elements = {'Ni': 0.25, 'Cr': 0.25, 'Mo': 0.25, 'W': 0.25} # 'comp_ntypes' kullanıcıdan alınabilir veya hesaplanabilir
        # comp_to_predict_custom = {k:v for k,v in custom_elements.items() if k in model_feature_cols}
        # if 'comp_ntypes' in model_feature_cols:
        #     element_keys_in_custom = [k for k in comp_to_predict_custom.keys() if k != 'comp_ntypes' and comp_to_predict_custom.get(k, 0) > 1e-6]
        #     comp_to_predict_custom['comp_ntypes'] = len(element_keys_in_custom) # Ya da kullanıcıdan alınır: custom_elements.get('comp_ntypes', len(element_keys_in_custom))

        # predict_delta_e_specific(
        #     composition_dict=comp_to_predict_custom,
        #     model=trained_model,
        #     all_feature_cols=model_feature_cols,
        #     scaler=trained_scaler
        # )

    else:
        print("\nModel, ölçekleyici veya özellikler yüklenemediği için tahmin başlatılamıyor.")