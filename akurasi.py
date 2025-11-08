import pandas as pd
from difflib import SequenceMatcher
import os
from sklearn.metrics import confusion_matrix
import numpy as np


BASE_DIR = '.' 
GOOGLE_SHEET_CSV = os.path.join(BASE_DIR, 'hasil_deteksi.csv') 
IMAGE_FOLDER = os.path.join(BASE_DIR, 'Foto_KTP') 


def calculate_similarity(a, b):
    """Menghitung skor kesamaan string menggunakan SequenceMatcher (0.0 hingga 1.0)"""
    return SequenceMatcher(None, str(a).lower().strip(), str(b).lower().strip()).ratio()

def check_accuracy(df):
    """
    Membandingkan kolom Deteksi dengan kolom Ground Truth dan Menghitung Semua Metrik.
    """
    
    field_mapping = {
        'NIK': ('NIK', 'NIK_Benar'),
        'NAMA': ('NAMA', 'Nama_Benar')
    }
    
    fields_to_check = list(field_mapping.keys())
    accuracy_results = []
    
    y_true_overall = []
    y_pred_overall = []

    for index, row in df.iterrows():
        file_name = row['Nama_file'] 
        total_fields = len(fields_to_check)
        correct_fields = 0
        field_accuracies = {}
        
        for field in fields_to_check:
            det_col, gt_col = field_mapping[field]
            
            gt_value = row[gt_col]
            det_value = row[det_col]
            
            is_exact_match = (str(gt_value).strip().lower() == str(det_value).strip().lower())
            
            if is_exact_match:
                correct_fields += 1
                field_accuracies[f'{field}_Match_Status'] = 'MATCH'
            else:
                similarity_score = calculate_similarity(gt_value, det_value)
                field_accuracies[f'{field}_Match_Status'] = f'MISMATCH (Sim: {similarity_score:.2f})'
                
        overall_accuracy = 1 if correct_fields == total_fields else 0
        field_accuracy_rate = correct_fields / total_fields
        
        y_true_overall.append(1) 
        y_pred_overall.append(overall_accuracy) 
        
        result = {
            'Nama_file': file_name,
            'Overall_KTP_Accuracy': overall_accuracy, 
            'Field_Accuracy_Rate': f'{field_accuracy_rate*100:.2f}%', 
            **field_accuracies
        }
        accuracy_results.append(result)

    results_df = pd.DataFrame(accuracy_results)
    
    cm = confusion_matrix(y_true_overall, y_pred_overall, labels=[0, 1])
 
    FN = cm[1, 0] # False Negative: Seharusnya Benar (1), diprediksi Salah (0) -> KTP GAGAL
    TP = cm[1, 1] # True Positive: Seharusnya Benar (1), diprediksi Benar (1) -> KTP BERHASIL
    
    # Hitung Metrik
    total_samples = len(results_df)
    Accuracy = TP / total_samples
    Precision = TP / (TP + 0) if (TP + 0) > 0 else 0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1_Score = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0
    
    
    print("\n--- Ringkasan Kinerja (Overall KTP Match) ---")
    print(f"Total Sampel KTP: {total_samples}")
    print("---------------------------------------------")
    print(f"✅ True Positives (KTP Benar): {TP}")
    print(f"❌ False Negatives (KTP Gagal): {FN}")
    print("---------------------------------------------")
    print(f"📊 Accuracy: {Accuracy * 100:.2f}%")
    print(f"📊 Precision: {Precision * 100:.2f}%")
    print(f"📊 Recall: {Recall * 100:.2f}%")
    print(f"📊 F1 Score: {F1_Score * 100:.2f}%")
    print("---------------------------------------------")
    
    print("\n📊 Matriks Kebingungan:")
    print("---------------------------------------------")
    print(f"| {'Predicted 0 (Gagal)':<20} | {'Predicted 1 (Berhasil)':<20} |")
    print("---------------------------------------------")
    print(f"| {'Actual 0 (TN=' + str(cm[0, 0]) + ')':<20} | {'Actual 0 (FP=' + str(cm[0, 1]) + ')':<20} |")
    print(f"| {'Actual 1 (FN=' + str(FN) + ')':<20} | {'Actual 1 (TP=' + str(TP) + ')':<20} |")
    print("---------------------------------------------")

    return results_df

try:
    df_data = pd.read_csv(GOOGLE_SHEET_CSV) 
    print(f"✅ Berhasil membaca data dari: {GOOGLE_SHEET_CSV}")
    accuracy_df = check_accuracy(df_data)
    output_filename = os.path.join(os.path.dirname(GOOGLE_SHEET_CSV), 'hasil_pengecekan_akurasi.csv')
    accuracy_df.to_csv(output_filename, index=False)
    print(f"✅ Hasil pengecekan akurasi telah disimpan ke: {output_filename}")
    print("\nBerikut adalah 5 baris pertama dari hasil:")
    print(accuracy_df.head())

except ImportError:
    print("❌ ERROR: Library 'scikit-learn' belum terinstal. Jalankan: pip install scikit-learn")
except FileNotFoundError as e:
    print(f"❌ ERROR: Pastikan file CSV 'hasil_deteksi.csv' berada di direktori yang sama dengan script ini. Error: {e}")
except Exception as e:
    print(f"❌ Terjadi kesalahan: {e}")