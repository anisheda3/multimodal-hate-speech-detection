"""
SIMPLE CODE TO FIND ALL COLUMNS IN DATASETS
"""

import os
import pandas as pd

def analyze_dataset_columns():
    """Analyze all datasets and print their columns"""
    
    base_dir = r"C:\Users\G ABHINAV REDDY\Downloads\processed_data"
    
    # Define dataset paths
    datasets = {
        'hate_speech_curated': os.path.join(base_dir, "hate_speech_curated"),
        'hate_speech_offensive': os.path.join(base_dir, "hate_speech_and_offensive_language"),
        'suspicious_comm': os.path.join(base_dir, "suspicious_communication_on_social_platforms"),
        'jigsaw_toxic': os.path.join(base_dir, "jigsaw-toxic-comment-classification-challenge"),
        'memotion_7k': os.path.join(base_dir, "memotion_dataset_7k")
    }
    
    def safe_read_csv(csv_file, nrows=5):
        """Safely read first few rows of CSV to get columns"""
        try:
            df = pd.read_csv(csv_file, encoding='utf-8', nrows=nrows, low_memory=False)
            return df
        except:
            try:
                df = pd.read_csv(csv_file, encoding='utf-8-sig', nrows=nrows, low_memory=False)
                return df
            except:
                try:
                    df = pd.read_csv(csv_file, encoding='latin-1', nrows=nrows, low_memory=False)
                    return df
                except:
                    return None
    
    def safe_read_excel(excel_file, nrows=5):
        """Safely read first few rows of Excel file"""
        try:
            df = pd.read_excel(excel_file, nrows=nrows)
            return df
        except:
            return None
    
    print("🔍 ANALYZING DATASET COLUMNS...")
    print("=" * 60)
    
    for dataset_name, dataset_path in datasets.items():
        print(f"\n📁 {dataset_name.upper()}")
        print("-" * 40)
        
        if not os.path.exists(dataset_path):
            print(f"❌ Directory not found: {dataset_path}")
            continue
        
        # Find all files in the directory
        all_files = []
        for file in os.listdir(dataset_path):
            if file.lower().endswith(('.csv', '.xlsx', '.xls', '.pickle', '.pkl')):
                all_files.append(os.path.join(dataset_path, file))
        
        if not all_files:
            print("   ⚠️ No data files found")
            continue
        
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            print(f"\n   📄 {file_name}")
            
            if file_path.lower().endswith('.csv'):
                df = safe_read_csv(file_path)
            elif file_path.lower().endswith(('.xlsx', '.xls')):
                df = safe_read_excel(file_path)
            elif file_path.lower().endswith(('.pickle', '.pkl')):
                try:
                    df = pd.read_pickle(file_path)
                    if len(df) > 5:
                        df = df.head(5)
                except:
                    df = None
            else:
                df = None
            
            if df is None or df.empty:
                print("      ❌ Could not read file")
                continue
            
            print(f"      📊 Shape: {df.shape}")
            print(f"      🏷️ Columns: {list(df.columns)}")
            
            # Show sample data types and first values
            print("      📋 Sample data:")
            for col in df.columns:
                sample_value = df[col].iloc[0] if len(df) > 0 else "N/A"
                dtype = df[col].dtype
                print(f"        - {col} ({dtype}): {str(sample_value)[:50]}...")
    
    print("\n" + "=" * 60)
    print("✅ COLUMN ANALYSIS COMPLETED!")

if __name__ == "__main__":
    analyze_dataset_columns()