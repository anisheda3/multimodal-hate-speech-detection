"""
TEXT DATASET PREPROCESSING - WITH SMART TEXT CLEANING
"""

import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split

def process_text_datasets():
    """Process text datasets with correct column mappings and smart text cleaning"""
    
    base_dir = r"C:\Users\G ABHINAV REDDY\Downloads\processed_data"
    processed_dir = os.path.join(base_dir, "processed_data")
    os.makedirs(processed_dir, exist_ok=True)
    
    all_datasets_processed = {}
    
    def smart_text_cleaning(text):
        """Smart cleaning that preserves important hate speech signals"""
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.lower().strip()
        text = ' '.join(text.split())
        
        # Remove URLs and emails (usually noise)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Normalize excessive punctuation (keep at least one)
        text = re.sub(r'!+', '!', text)  # "!!!!" → "!"
        text = re.sub(r'\?+', '?', text) # "????" → "?"
        text = re.sub(r'\.+', '.', text) # "......" → "."
        
        # Remove extra whitespace again
        text = ' '.join(text.split())
        
        return text
    
    def safe_read_csv(csv_file, nrows=None):
        """Safe CSV reading with multiple encodings and better error handling"""
        try:
            df = pd.read_csv(csv_file, encoding='utf-8', on_bad_lines='skip', low_memory=False, nrows=nrows)
            return df
        except:
            try:
                df = pd.read_csv(csv_file, encoding='utf-8-sig', on_bad_lines='skip', low_memory=False, nrows=nrows)
                return df
            except:
                try:
                    df = pd.read_csv(csv_file, encoding='latin-1', on_bad_lines='skip', low_memory=False, nrows=nrows)
                    return df
                except:
                    print(f"    ❌ Could not read: {csv_file}")
                    return pd.DataFrame()
    
    def safe_int_convert(value):
        """Safely convert value to integer"""
        try:
            return int(float(value))
        except:
            return 0
    
    def process_hate_speech_curated():
        """Process hate_speech_curated dataset"""
        print(f"\n🔄 PROCESSING: hate_speech_curated")
        
        dataset_path = os.path.join(base_dir, "hate_speech_curated")
        all_data = []
        
        # Process HateSpeechDataset.csv
        file1 = os.path.join(dataset_path, "HateSpeechDataset.csv")
        if os.path.exists(file1):
            print(f"  Reading: HateSpeechDataset.csv")
            df = safe_read_csv(file1)
            if not df.empty:
                print(f"    Columns: {list(df.columns)}")
                print(f"    First few rows sample:")
                for i in range(min(3, len(df))):
                    print(f"      Row {i}: Content='{str(df.iloc[i]['Content'])[:50]}...', Label='{df.iloc[i]['Label']}'")
                
                valid_count = 0
                for idx, row in df.iterrows():
                    try:
                        text = str(row['Content']) if pd.notna(row['Content']) else ""
                        label_val = row['Label']
                        
                        # Skip header rows or invalid data
                        if text.lower() in ['content', 'label', 'content_int'] or text == '':
                            continue
                            
                        text = smart_text_cleaning(text)
                        if not text or len(text) < 5:
                            continue
                        
                        label = safe_int_convert(label_val)
                        
                        all_data.append({
                            'text': text,
                            'label': label,
                            'source_dataset': 'hate_speech_curated',
                            'source_file': 'HateSpeechDataset.csv'
                        })
                        valid_count += 1
                        
                    except Exception as e:
                        continue
                
                print(f"    Processed {valid_count} valid samples from HateSpeechDataset.csv")
        
        # Process HateSpeechDatasetBalanced.csv
        file2 = os.path.join(dataset_path, "HateSpeechDatasetBalanced.csv")
        if os.path.exists(file2):
            print(f"  Reading: HateSpeechDatasetBalanced.csv")
            df = safe_read_csv(file2)
            if not df.empty:
                print(f"    Columns: {list(df.columns)}")
                print(f"    First few rows sample:")
                for i in range(min(3, len(df))):
                    print(f"      Row {i}: Content='{str(df.iloc[i]['Content'])[:50]}...', Label='{df.iloc[i]['Label']}'")
                
                valid_count = 0
                for idx, row in df.iterrows():
                    try:
                        text = str(row['Content']) if pd.notna(row['Content']) else ""
                        label_val = row['Label']
                        
                        # Skip header rows or invalid data
                        if text.lower() in ['content', 'label'] or text == '':
                            continue
                            
                        text = smart_text_cleaning(text)
                        if not text or len(text) < 5:
                            continue
                        
                        label = safe_int_convert(label_val)
                        
                        all_data.append({
                            'text': text,
                            'label': label,
                            'source_dataset': 'hate_speech_curated',
                            'source_file': 'HateSpeechDatasetBalanced.csv'
                        })
                        valid_count += 1
                        
                    except Exception as e:
                        continue
                
                print(f"    Processed {valid_count} valid samples from HateSpeechDatasetBalanced.csv")
        
        if all_data:
            result_df = pd.DataFrame(all_data)
            result_df = result_df.drop_duplicates(subset=['text'])
            output_path = os.path.join(processed_dir, "hate_speech_curated_cleaned.csv")
            result_df.to_csv(output_path, index=False)
            
            hate_count = result_df['label'].sum()
            all_datasets_processed['hate_speech_curated'] = {
                'samples': len(result_df),
                'hate_samples': hate_count
            }
            print(f"✅ hate_speech_curated: {len(result_df)} samples ({hate_count} hate)")
        
        return all_data
    
    def process_hate_speech_offensive():
        """Process hate_speech_offensive dataset"""
        print(f"\n🔄 PROCESSING: hate_speech_offensive")
        
        dataset_path = os.path.join(base_dir, "hate_speech_and_offensive_language")
        all_data = []
        
        file_path = os.path.join(dataset_path, "labeled_data.csv")
        if os.path.exists(file_path):
            print(f"  Reading: labeled_data.csv")
            df = safe_read_csv(file_path)
            if not df.empty:
                print(f"    Columns: {list(df.columns)}")
                print(f"    First few rows sample:")
                for i in range(min(3, len(df))):
                    print(f"      Row {i}: tweet='{str(df.iloc[i]['tweet'])[:50]}...', class='{df.iloc[i]['class']}'")
                
                valid_count = 0
                # In this dataset: class 0 = hate speech, class 1 = offensive language, class 2 = neither
                # We'll consider both hate speech (0) and offensive language (1) as "hate"
                for idx, row in df.iterrows():
                    try:
                        text = str(row['tweet']) if pd.notna(row['tweet']) else ""
                        class_val = row['class']
                        
                        # Skip header rows or invalid data
                        if text.lower() in ['tweet', 'class'] or text == '':
                            continue
                            
                        text = smart_text_cleaning(text)
                        if not text or len(text) < 5:
                            continue
                        
                        # Convert: class 0 or 1 -> hate (1), class 2 -> not hate (0)
                        label = 1 if safe_int_convert(class_val) in [0, 1] else 0
                        
                        all_data.append({
                            'text': text,
                            'label': label,
                            'source_dataset': 'hate_speech_offensive',
                            'source_file': 'labeled_data.csv',
                            'original_class': safe_int_convert(class_val)
                        })
                        valid_count += 1
                        
                    except Exception as e:
                        continue
                
                print(f"    Processed {valid_count} samples from labeled_data.csv")
        
        if all_data:
            result_df = pd.DataFrame(all_data)
            result_df = result_df.drop_duplicates(subset=['text'])
            output_path = os.path.join(processed_dir, "hate_speech_offensive_cleaned.csv")
            result_df.to_csv(output_path, index=False)
            
            hate_count = result_df['label'].sum()
            all_datasets_processed['hate_speech_offensive'] = {
                'samples': len(result_df),
                'hate_samples': hate_count
            }
            print(f"✅ hate_speech_offensive: {len(result_df)} samples ({hate_count} hate)")
        
        return all_data
    
    def process_suspicious_comm():
        """Process suspicious_comm dataset"""
        print(f"\n🔄 PROCESSING: suspicious_comm")
        
        dataset_path = os.path.join(base_dir, "suspicious_communication_on_social_platforms")
        all_data = []
        
        file_path = os.path.join(dataset_path, "Suspicious Communication on Social Platforms.csv")
        if os.path.exists(file_path):
            print(f"  Reading: Suspicious Communication on Social Platforms.csv")
            df = safe_read_csv(file_path)
            if not df.empty:
                print(f"    Columns: {list(df.columns)}")
                print(f"    First few rows sample:")
                for i in range(min(3, len(df))):
                    print(f"      Row {i}: comments='{str(df.iloc[i]['comments'])[:50]}...', tagging='{df.iloc[i]['tagging']}'")
                
                valid_count = 0
                for idx, row in df.iterrows():
                    try:
                        text = str(row['comments']) if pd.notna(row['comments']) else ""
                        tagging_val = row['tagging']
                        
                        # Skip header rows or invalid data
                        if text.lower() in ['comments', 'tagging'] or text == '':
                            continue
                            
                        text = smart_text_cleaning(text)
                        if not text or len(text) < 5:
                            continue
                        
                        label = safe_int_convert(tagging_val)
                        
                        all_data.append({
                            'text': text,
                            'label': label,
                            'source_dataset': 'suspicious_comm',
                            'source_file': 'Suspicious Communication on Social Platforms.csv'
                        })
                        valid_count += 1
                        
                    except Exception as e:
                        continue
                
                print(f"    Processed {valid_count} samples from Suspicious Communication on Social Platforms.csv")
        
        if all_data:
            result_df = pd.DataFrame(all_data)
            result_df = result_df.drop_duplicates(subset=['text'])
            output_path = os.path.join(processed_dir, "suspicious_comm_cleaned.csv")
            result_df.to_csv(output_path, index=False)
            
            hate_count = result_df['label'].sum()
            all_datasets_processed['suspicious_comm'] = {
                'samples': len(result_df),
                'hate_samples': hate_count
            }
            print(f"✅ suspicious_comm: {len(result_df)} samples ({hate_count} hate)")
        
        return all_data
    
    def process_jigsaw_toxic():
        """Process jigsaw toxic dataset"""
        print(f"\n🔄 PROCESSING: jigsaw_toxic")
        
        dataset_path = os.path.join(base_dir, "jigsaw-toxic-comment-classification-challenge")
        all_data = []
        
        # Process train.csv
        train_file = os.path.join(dataset_path, "train.csv")
        if os.path.exists(train_file):
            print(f"  Reading: train.csv")
            df = safe_read_csv(train_file)
            if not df.empty:
                print(f"    Columns: {list(df.columns)}")
                
                toxic_count = 0
                valid_count = 0
                for idx, row in df.iterrows():
                    try:
                        text = str(row['comment_text']) if pd.notna(row['comment_text']) else ""
                        
                        # Skip header rows or invalid data
                        if text.lower() in ['comment_text', 'id'] or text == '':
                            continue
                            
                        text = smart_text_cleaning(text)
                        if not text or len(text) < 5:
                            continue
                        
                        # Check if any toxicity label is 1
                        toxic_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
                        is_toxic = any(safe_int_convert(row[label]) == 1 for label in toxic_labels if pd.notna(row[label]))
                        
                        if is_toxic:
                            toxic_count += 1
                        
                        all_data.append({
                            'text': text,
                            'label': 1 if is_toxic else 0,
                            'source_dataset': 'jigsaw_toxic',
                            'source_file': 'train.csv'
                        })
                        valid_count += 1
                        
                    except Exception as e:
                        continue
                
                print(f"    Processed {valid_count} samples from train.csv ({toxic_count} toxic)")
        
        # Process test.csv with test_labels.csv
        test_file = os.path.join(dataset_path, "test.csv")
        test_labels_file = os.path.join(dataset_path, "test_labels.csv")
        
        if os.path.exists(test_file) and os.path.exists(test_labels_file):
            print(f"  Reading: test.csv with test_labels.csv")
            test_df = safe_read_csv(test_file)
            test_labels_df = safe_read_csv(test_labels_file)
            
            if not test_df.empty and not test_labels_df.empty:
                # Merge test data with labels
                merged_df = pd.merge(test_df, test_labels_df, on='id', how='inner')
                toxic_count = 0
                valid_count = 0
                
                for idx, row in merged_df.iterrows():
                    try:
                        text = str(row['comment_text']) if pd.notna(row['comment_text']) else ""
                        
                        # Skip header rows or invalid data
                        if text.lower() in ['comment_text', 'id'] or text == '':
                            continue
                            
                        # Skip if toxic label is -1 (not labeled)
                        if row.get('toxic', 0) == -1:
                            continue
                        
                        text = smart_text_cleaning(text)
                        if not text or len(text) < 5:
                            continue
                        
                        # Check if any toxicity label is 1
                        toxic_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
                        is_toxic = any(safe_int_convert(row[label]) == 1 for label in toxic_labels if pd.notna(row[label]) and row[label] != -1)
                        
                        if is_toxic:
                            toxic_count += 1
                        
                        all_data.append({
                            'text': text,
                            'label': 1 if is_toxic else 0,
                            'source_dataset': 'jigsaw_toxic',
                            'source_file': 'test.csv'
                        })
                        valid_count += 1
                        
                    except Exception as e:
                        continue
                
                print(f"    Processed {valid_count} samples from test data ({toxic_count} toxic)")
        
        if all_data:
            result_df = pd.DataFrame(all_data)
            result_df = result_df.drop_duplicates(subset=['text'])
            output_path = os.path.join(processed_dir, "jigsaw_toxic_cleaned.csv")
            result_df.to_csv(output_path, index=False)
            
            hate_count = result_df['label'].sum()
            all_datasets_processed['jigsaw_toxic'] = {
                'samples': len(result_df),
                'hate_samples': hate_count
            }
            print(f"✅ jigsaw_toxic: {len(result_df)} samples ({hate_count} hate)")
        
        return all_data
    
    # Process all datasets
    print("🚀 PROCESSING TEXT DATASETS WITH SMART CLEANING...")
    
    process_hate_speech_curated()
    process_hate_speech_offensive()
    process_suspicious_comm()
    process_jigsaw_toxic()
    
    # Create combined dataset
    if all_datasets_processed:
        print(f"\n📊 CREATING COMBINED DATASET...")
        
        all_dfs = []
        for dataset_name in all_datasets_processed.keys():
            try:
                csv_path = os.path.join(processed_dir, f"{dataset_name}_cleaned.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    df['dataset_source'] = dataset_name
                    all_dfs.append(df)
                    print(f"  Added {dataset_name}: {len(df)} samples")
            except Exception as e:
                print(f"Error reading {dataset_name}: {e}")
                continue
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['text'])
            
            combined_path = os.path.join(processed_dir, "ALL_TEXT_DATASETS_COMBINED.csv")
            combined_df.to_csv(combined_path, index=False)
            
            # Create train/val/test splits
            train_df, temp_df = train_test_split(
                combined_df, test_size=0.3, stratify=combined_df['label'], random_state=42
            )
            val_df, test_df = train_test_split(
                temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
            )
            
            # Create split directories
            splits_dir = os.path.join(processed_dir, "splits")
            os.makedirs(os.path.join(splits_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(splits_dir, "val"), exist_ok=True)
            os.makedirs(os.path.join(splits_dir, "test"), exist_ok=True)
            
            # Save splits
            train_df.to_csv(os.path.join(splits_dir, "train", "text_train.csv"), index=False)
            val_df.to_csv(os.path.join(splits_dir, "val", "text_val.csv"), index=False)
            test_df.to_csv(os.path.join(splits_dir, "test", "text_test.csv"), index=False)
            
            # Final summary
            total_samples = len(combined_df)
            total_hate = combined_df['label'].sum()
            
            print(f"""
🎉 TEXT PROCESSING COMPLETED!
=============================

📊 FINAL SUMMARY:
- Total datasets: {len(all_datasets_processed)}
- Total samples: {total_samples:,}
- Hate speech: {total_hate:,}
- Normal: {total_samples - total_hate:,}
- Hate ratio: {(total_hate/total_samples)*100:.1f}%

📁 SPLIT DISTRIBUTION:
- Train: {len(train_df):,} samples
- Validation: {len(val_df):,} samples  
- Test: {len(test_df):,} samples

📂 OUTPUT FILES:
- Combined dataset: {combined_path}
- Splits: {splits_dir}/
- Individual datasets: {processed_dir}/*_cleaned.csv

📊 DATASET BREAKDOWN:""")
            
            for dataset_name, info in all_datasets_processed.items():
                ratio = (info['hate_samples']/info['samples'])*100 if info['samples'] > 0 else 0
                print(f"- {dataset_name}: {info['samples']:,} samples ({info['hate_samples']:,} hate, {ratio:.1f}%)")
    
    else:
        print("❌ No datasets were successfully processed!")

if __name__ == "__main__":
    process_text_datasets()