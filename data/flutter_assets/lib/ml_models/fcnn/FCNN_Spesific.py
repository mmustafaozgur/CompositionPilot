# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib # Scaler ve özellik listesini kaydetmek/yüklemek için
import tensorflow as tf
from tensorflow.keras.models import load_model # Model yükleme için
import os

#-------------------------------------------------------------------------------
# Blok 7'den Fonksiyon (Kullanıcı tarafından sağlandı - predict_delta_e_specific_mlp)
#-------------------------------------------------------------------------------
def predict_delta_e_specific_mlp(composition_dict, model, all_feature_cols, scaler, fixed_external_features=None):
    """
    Verilen spesifik bir kompozisyon ve harici sabit özellikler için MLP modeli ile Delta E değerini tahmin eder.
    composition_dict: {'ElementSembolü': fraksiyon, ...} SADECE element fraksiyonlarını içermeli.
    fixed_external_features: {'feature_name': value} örn: {'comp_ntypes': 3, 'temperature': 300}
                                 Bu özellikler composition_dict'te OLMAMALI, ayrı verilmeli.
    """
    print(f"\nSpesifik kompozisyon için MLP ile tahmin (elementler): {composition_dict}")
    if fixed_external_features:
        print(f"Kullanılan sabit/harici özellikler: {fixed_external_features}")

    # Modelin eğitildiği tüm özellikleri içeren bir sözlük oluştur ve başlangıçta sıfırla
    input_data_dict = {feature: 0.0 for feature in all_feature_cols}

    # Element fraksiyonlarını input_data_dict'e aktar
    active_element_keys_in_comp = []
    for elem, frac in composition_dict.items():
        if elem in input_data_dict:
            input_data_dict[elem] = frac
            if frac > 1e-6: # Çok küçük fraksiyonları aktif sayma
                active_element_keys_in_comp.append(elem)
        else:
            print(f"Uyarı: Element '{elem}' modelin özellik listesinde bulunmuyor ve atlanacak.")

    # Sabit/harici özellikleri input_data_dict'e aktar
    if fixed_external_features:
        for feat, val in fixed_external_features.items():
            if feat in input_data_dict:
                input_data_dict[feat] = val
            else:
                print(f"Uyarı: Sabit özellik '{feat}' modelin özellik listesinde bulunmuyor ve atlanacak.")

    # comp_ntypes'ı kontrol et ve gerekirse hesapla
    # Bu özellik model tarafından kullanılıyorsa ve harici olarak belirtilmediyse hesaplanır.
    if 'comp_ntypes' in input_data_dict:
        if fixed_external_features and 'comp_ntypes' in fixed_external_features:
            # Kullanıcı tarafından harici olarak verildiyse onu kullan
            pass
        else:
            # Harici olarak verilmediyse, aktif element sayısına göre ayarla
            # (Yalnızca element fraksiyonları verilen ve comp_ntypes harici olarak belirtilmeyen durumlar için)
            input_data_dict['comp_ntypes'] = len(active_element_keys_in_comp)
            print(f"'comp_ntypes' aktif element sayısına göre otomatik olarak {len(active_element_keys_in_comp)} değerine ayarlandı.")

    # Tahmin için en az bir geçerli özellik olup olmadığını kontrol et
    # (Yalnızca sıfır olan elementler veya sadece harici özellikler girilmiş olabilir)
    is_any_feature_set = False
    if active_element_keys_in_comp: # Aktif element var mı?
        is_any_feature_set = True
    if not is_any_feature_set and fixed_external_features: # Aktif element yoksa, harici özellik var mı?
        if any(feat in input_data_dict and fixed_external_features[feat] != 0.0 for feat in fixed_external_features):
            is_any_feature_set = True
            
    if not is_any_feature_set:
        print("HATA: Verilen kompozisyonda model tarafından tanınan aktif element veya anlamlı sabit özellik yok. Tahmin yapılamıyor.")
        return None

    # Tahmin için DataFrame oluştur
    input_df = pd.DataFrame([input_data_dict], columns=all_feature_cols)

    # Veriyi ölçekle
    input_df_scaled = scaler.transform(input_df)

    try:
        # Tahmin yap
        predicted_delta_e = model.predict(input_df_scaled).flatten() # MLP için .flatten()
        print(f"Tahmin Edilen Delta E (MLP): {predicted_delta_e[0]:.4f}")
        return predicted_delta_e[0] if predicted_delta_e is not None else None
    except Exception as e:
        print(f"Spesifik kompozisyon için MLP ile tahmin sırasında hata: {e}")
        return None

#-------------------------------------------------------------------------------
# Ana Çalıştırma Bloğu
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    # Notebook'taki train_evaluate_save_mlp_model fonksiyonunun
    # feature_selection_method='lasso' ve base_filename='mlp_delta_e_model'
    # ile ürettiği dosya adları
    MODEL_FILENAME = 'mlp_delta_e_model_lasso.h5'
    SCALER_FILENAME = 'mlp_scaler_lasso.joblib'
    FEATURES_FILENAME = 'mlp_model_features_lasso.joblib'

    trained_mlp_model = None
    trained_final_scaler = None
    selected_model_features = None

    # Model, ölçekleyici ve özellik listesini yükle
    try:
        print(f"MLP Modeli '{MODEL_FILENAME}' dosyasından yükleniyor...")
        if not os.path.exists(MODEL_FILENAME):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_FILENAME}")
        trained_mlp_model = load_model(MODEL_FILENAME)
        print("MLP Modeli yüklendi.")
        
        print(f"Ölçekleyici '{SCALER_FILENAME}' dosyasından yükleniyor...")
        if not os.path.exists(SCALER_FILENAME):
            raise FileNotFoundError(f"Ölçekleyici dosyası bulunamadı: {SCALER_FILENAME}")
        trained_final_scaler = joblib.load(SCALER_FILENAME)
        print("Ölçekleyici yüklendi.")
        
        print(f"Model özellikleri '{FEATURES_FILENAME}' dosyasından yükleniyor...")
        if not os.path.exists(FEATURES_FILENAME):
            raise FileNotFoundError(f"Özellik dosyası bulunamadı: {FEATURES_FILENAME}")
        selected_model_features = joblib.load(FEATURES_FILENAME)
        print(f"Model özellikleri yüklendi. Toplam {len(selected_model_features)} özellik.")
        
    except FileNotFoundError as e:
        print(f"HATA: Gerekli dosya bulunamadı: {e}.")
        print("Lütfen modelin, ölçekleyicinin ve özellik dosyasının doğru yolda olduğundan ve daha önce oluşturulduğundan emin olun.")
        exit()
    except Exception as e:
        print(f"Dosyalar yüklenirken bir hata oluştu: {e}")
        exit()

    # Yükleme başarılıysa tahmin yap
    if trained_mlp_model and trained_final_scaler and selected_model_features:
        print("\n--- Spesifik Kompozisyonlar İçin Delta E Tahmini (MLP ile) Başlatılıyor ---")
        
        # Örnek 1: Sadece element fraksiyonları ile
        # (comp_ntypes otomatik hesaplanacak)
        specific_composition_1_elements_only = {'Mo': 0.6, 'Ho': 0.2, 'Ga': 0.1, 'U': 0.1}
        # Element olmayan, sabit/harici özellikler (bu örnekte yok ama gerekirse eklenebilir)
        fixed_ext_features_1 = {} 
        
        # Modelin bildiği elementleri filtrele
        final_specific_comp_1_for_prediction = {
            k: v for k, v in specific_composition_1_elements_only.items() if k in selected_model_features
        }
        
        if final_specific_comp_1_for_prediction or (fixed_ext_features_1 and any(feat in selected_model_features for feat in fixed_ext_features_1)):
            predict_delta_e_specific_mlp(
                composition_dict=final_specific_comp_1_for_prediction,
                model=trained_mlp_model,
                all_feature_cols=selected_model_features, # Bu, yüklenen özellik listesi olmalı
                scaler=trained_final_scaler,
                fixed_external_features=fixed_ext_features_1
            )
        else:
            print(f"\nÖrnek 1 için '{specific_composition_1_elements_only}' kompozisyonunda modelde tanımlı element bulunamadı veya zorunlu sabit özellik eksik, tahmin atlanıyor.")

        print("-" * 50)

        # Örnek 2: Element fraksiyonları ve harici olarak belirtilmiş 'comp_ntypes' ile
        specific_composition_2_elements_only = {'Fe': 0.5, 'Cr': 0.3, 'Ni': 0.2}
        # 'comp_ntypes' harici olarak veriliyor.
        fixed_ext_features_2 = {'comp_ntypes': 3} 
        # Eğer 'temperature' gibi başka sabit özellikler modelde varsa, onlar da buraya eklenebilir.
        # Örneğin: fixed_ext_features_2 = {'comp_ntypes': 3, 'temperature': 300}

        final_specific_comp_2_for_prediction = {
            k: v for k, v in specific_composition_2_elements_only.items() if k in selected_model_features
        }
        
        if final_specific_comp_2_for_prediction or (fixed_ext_features_2 and any(feat in selected_model_features for feat in fixed_ext_features_2)):
            predict_delta_e_specific_mlp(
                composition_dict=final_specific_comp_2_for_prediction,
                model=trained_mlp_model,
                all_feature_cols=selected_model_features,
                scaler=trained_final_scaler,
                fixed_external_features=fixed_ext_features_2
            )
        else:
            print(f"\nÖrnek 2 için '{specific_composition_2_elements_only}' kompozisyonunda modelde tanımlı element bulunamadı veya zorunlu sabit özellik eksik, tahmin atlanıyor.")
            
        print("-" * 50)

        # Örnek 3: Modelin tanımadığı bir element ile
        specific_composition_3_elements_only = {'Xy': 0.5, 'Zr': 0.5} # Xy modelde olmayabilir
        fixed_ext_features_3 = {}
        
        final_specific_comp_3_for_prediction = {
            k: v for k, v in specific_composition_3_elements_only.items() if k in selected_model_features
        }
        
        if final_specific_comp_3_for_prediction or (fixed_ext_features_3 and any(feat in selected_model_features for feat in fixed_ext_features_3)):
             predict_delta_e_specific_mlp(
                composition_dict=final_specific_comp_3_for_prediction,
                model=trained_mlp_model,
                all_feature_cols=selected_model_features,
                scaler=trained_final_scaler,
                fixed_external_features=fixed_ext_features_3
            )
        else:
            print(f"\nÖrnek 3 için '{specific_composition_3_elements_only}' kompozisyonunda modelde tanımlı element bulunamadı veya zorunlu sabit özellik eksik, tahmin atlanıyor.")


        print("-" * 50)
        
        # Örnek 4: Hiç element olmayan, sadece harici özelliklerle (eğer model böyle eğitildiyse)
        # Bu örnek, modelinizin element dışı özelliklerle de tahmin yapıp yapamadığına bağlıdır.
        # Eğer model sadece element fraksiyonları ve comp_ntypes ile eğitildiyse, bu örnek anlamlı olmayabilir.
        specific_composition_4_elements_only = {} 
        fixed_ext_features_4 = {'comp_ntypes': 3, 'Fe': 0.5, 'Sb': 0.25, 'Ni': 0.25} # Sadece 'Mo' elementi ve 1 tip element var gibi
                                                            # Ya da {'temperature': 500, 'pressure': 1} gibi tamamen harici özellikler
                                                            # Bu özelliklerin `selected_model_features` içinde olması gerekir.

        final_specific_comp_4_for_prediction = {
             k: v for k, v in specific_composition_4_elements_only.items() if k in selected_model_features
        }
        # fixed_external_features'daki özelliklerin de modelde olup olmadığını kontrol et
        final_fixed_ext_features_4 = {
            k: v for k,v in fixed_ext_features_4.items() if k in selected_model_features
        }


        if final_specific_comp_4_for_prediction or (final_fixed_ext_features_4 and any(feat in selected_model_features for feat in final_fixed_ext_features_4)):
            predict_delta_e_specific_mlp(
                composition_dict=final_specific_comp_4_for_prediction,
                model=trained_mlp_model,
                all_feature_cols=selected_model_features,
                scaler=trained_final_scaler,
                fixed_external_features=final_fixed_ext_features_4
            )
        else:
            print(f"\nÖrnek 4 için '{specific_composition_4_elements_only}' ve '{fixed_ext_features_4}' ile modelde tanımlı özellik bulunamadı, tahmin atlanıyor.")

    else:
        print("\nMLP Modeli, ölçekleyici veya özellikler yüklenemediği için tahmin başlatılamıyor.")