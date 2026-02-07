"""
MEMOTION DATASET PREPROCESSING - WITH FIXED REPORT AND DOCUMENTATION
"""

import os
import pandas as pd
import re
from PIL import Image
import glob

def process_memotion_dataset():
    """Process Memotion 7K dataset with corrected label mappings and proper reporting"""
    
    base_dir = r"C:\Users\G ABHINAV REDDY\Downloads\processed_data"
    memotion_dir = os.path.join(base_dir, "memotion_dataset_7k")
    processed_dir = os.path.join(base_dir, "processed_data")
    os.makedirs(processed_dir, exist_ok=True)
    
    print("🖼️ PROCESSING MEMOTION 7K DATASET...")
    
    if not os.path.exists(memotion_dir):
        print(f"❌ Memotion dataset not found: {memotion_dir}")
        return
    
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
    
    def load_labels():
        """Load label data from CSV or Excel"""
        labels_file = os.path.join(memotion_dir, "labels.csv")
        if os.path.exists(labels_file):
            print(f"📖 Loading labels from: labels.csv")
            df = pd.read_csv(labels_file)
        else:
            labels_file = os.path.join(memotion_dir, "labels.xlsx")
            if os.path.exists(labels_file):
                print(f"📖 Loading labels from: labels.xlsx")
                df = pd.read_excel(labels_file)
            else:
                print("❌ No labels file found")
                return None
        
        print(f"   Loaded {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
        return df
    
    def find_image_files():
        """Find all image files"""
        images_dir = os.path.join(memotion_dir, "images")
        if not os.path.exists(images_dir):
            print(f"❌ Images directory not found: {images_dir}")
            return {}
        
        image_files = {}
        supported_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        
        for format in supported_formats:
            pattern = os.path.join(images_dir, '**', format)
            for img_path in glob.glob(pattern, recursive=True):
                filename = os.path.basename(img_path)
                image_files[filename] = img_path
        
        print(f"📷 Found {len(image_files)} image files")
        return image_files
    
    def get_offensive_label(offensive_str):
        """
        Convert offensive string to binary label
        
        Memotion offensive categories:
        - 'not_offensive' → 0 (Not Hate) - No offensive content
        - 'slight' → 0 (Not Hate) - Mild/borderline offensive, not severe enough for hate speech
        - 'offensive' → 1 (Hate) - Clearly offensive content
        - 'very_offensive' → 1 (Hate) - Highly offensive content  
        - 'hateful_offensive' → 1 (Hate) - Hate speech specifically targeting groups
        """
        if not isinstance(offensive_str, str):
            return 0
        
        offensive_str = offensive_str.lower().strip()
        
        # Hate speech categories
        if offensive_str in ['offensive', 'very_offensive', 'hateful_offensive']:
            return 1
        # Not hate categories
        elif offensive_str in ['not_offensive', 'slight']:
            return 0
        # Default to not hate for unknown values
        else:
            return 0
    
    # Main processing
    labels_df = load_labels()
    if labels_df is None:
        return
    
    image_files = find_image_files()
    
    all_data = []
    processed_count = 0
    image_found_count = 0
    hate_count = 0
    
    print(f"\n🎯 PROCESSING {len(labels_df)} SAMPLES...")
    
    for idx, row in labels_df.iterrows():
        try:
            # Use text_corrected if available, otherwise text_ocr
            text = ""
            if pd.notna(row['text_corrected']):
                text = str(row['text_corrected'])
            elif pd.notna(row['text_ocr']):
                text = str(row['text_ocr'])
            else:
                continue
            
            # Skip header rows or invalid data
            if text.lower() in ['text_corrected', 'text_ocr', 'image_name'] or text == '':
                continue
            
            # Clean text using smart cleaning
            text = smart_text_cleaning(text)
            if not text or len(text) < 5:
                continue
            
            # Get offensive label (main hate speech indicator)
            offensive_str = row['offensive'] if pd.notna(row['offensive']) else "not_offensive"
            label = get_offensive_label(offensive_str)
            
            if label == 1:
                hate_count += 1
            
            # Find corresponding image
            image_name = str(row['image_name']) if pd.notna(row['image_name']) else ""
            image_path = None
            
            if image_name and image_name in image_files:
                image_path = image_files[image_name]
                # Validate image
                try:
                    with Image.open(image_path) as img:
                        img.verify()
                    image_found_count += 1
                except:
                    image_path = None
                    print(f"    ⚠️  Corrupted image: {image_name}")
            
            all_data.append({
                'text': text,
                'label': label,
                'image_path': image_path if image_path else "",
                'image_name': image_name,
                'source_dataset': 'memotion_7k',
                'offensive_category': offensive_str,
                'humour': row['humour'] if pd.notna(row['humour']) else "",
                'sarcasm': row['sarcasm'] if pd.notna(row['sarcasm']) else "",
                'motivational': row['motivational'] if pd.notna(row['motivational']) else "",
                'overall_sentiment': row['overall_sentiment'] if pd.notna(row['overall_sentiment']) else "",
                'has_image': 1 if image_path else 0
            })
            
            processed_count += 1
            
            if processed_count % 1000 == 0:
                print(f"  Processed {processed_count}/{len(labels_df)} samples... ({hate_count} hate so far)")
                
        except Exception as e:
            continue
    
    # Create final dataset
    if all_data:
        memotion_df = pd.DataFrame(all_data)
        
        # Remove duplicates based on text and image
        initial_count = len(memotion_df)
        memotion_df = memotion_df.drop_duplicates(subset=['text', 'image_name'])
        final_count = len(memotion_df)
        
        # Save the dataset
        output_path = os.path.join(processed_dir, "memotion_7k_multimodal.csv")
        memotion_df.to_csv(output_path, index=False)
        
        # Calculate statistics
        total_samples = len(memotion_df)
        total_hate = memotion_df['label'].sum()
        images_available = memotion_df['has_image'].sum()
        
        # Analyze offensive category distribution
        offensive_dist = memotion_df['offensive_category'].value_counts()
        
        print(f"""
🎉 MEMOTION PROCESSING COMPLETED!
=================================

📊 DATASET SUMMARY:
- Total samples: {total_samples:,}
- Hate speech samples: {total_hate:,}
- Normal samples: {total_samples - total_hate:,}
- Hate ratio: {(total_hate/total_samples)*100:.1f}%
- Images available: {images_available:,} ({images_available/total_samples*100:.1f}%)

📈 OFFENSIVE CATEGORY DISTRIBUTION:""")
        
        for category, count in offensive_dist.items():
            percentage = (count / total_samples) * 100
            hate_label = "HATE" if get_offensive_label(category) == 1 else "NOT HATE"
            print(f"  - {category}: {count:,} ({percentage:.1f}%) [{hate_label}]")

        print(f"""
📍 OUTPUT:
- Dataset: {output_path}

📁 FILES PROCESSED:
- Labels: {len(labels_df)} rows
- Images found: {image_found_count} files
- Valid multimodal pairs: {total_samples}
- Duplicates removed: {initial_count - final_count:,}
""")
        
        # ✅ FIXED: Save detailed analysis report
        report_path = os.path.join(processed_dir, "memotion_analysis_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"""Memotion 7K Dataset Analysis Report
=====================================

Dataset Location: {memotion_dir}
Processing Date: {pd.Timestamp.now()}

FINAL STATISTICS:
- Total samples processed: {total_samples}
- Hate speech samples: {total_hate}
- Normal samples: {total_samples - total_hate}
- Hate speech ratio: {(total_hate/total_samples)*100:.1f}%
- Images available: {images_available} ({images_available/total_samples*100:.1f}%)

OFFENSIVE CATEGORY BREAKDOWN:
""")
            for category, count in offensive_dist.items():
                percentage = (count / total_samples) * 100
                hate_label = "HATE" if get_offensive_label(category) == 1 else "NOT HATE"
                f.write(f"- {category}: {count} samples ({percentage:.1f}%) [{hate_label}]\n")
            
            f.write(f"""
PROCESSING DETAILS:
- Original label rows: {len(labels_df)}
- Image files found: {len(image_files)}
- Successful pairs: {processed_count}
- Images successfully matched: {image_found_count}
- Duplicates removed: {initial_count - final_count}

LABELING LOGIC EXPLANATION:
- HATE (1): 'offensive', 'very_offensive', 'hateful_offensive'
  * Clear hate speech, offensive content targeting groups

- NOT HATE (0): 'not_offensive', 'slight'
  * 'not_offensive': No offensive content
  * 'slight': Mild/borderline offensive (insults, subtle prejudice) but not severe hate speech

CATEGORY DEFINITIONS:
- not_offensive: No offensive content whatsoever
- slight: Mild offensive content, insults, borderline comments
- offensive: Clearly offensive content, profanity, strong insults  
- very_offensive: Highly offensive content, severe language
- hateful_offensive: Hate speech specifically targeting racial, religious, or other groups

ADDITIONAL FEATURES:
- humour: Categorical humor classification
- sarcasm: Categorical sarcasm classification  
- motivational: Motivational content flag
- overall_sentiment: Overall sentiment classification

OUTPUT FILES:
- Main dataset: {output_path}
- This report: {report_path}
""")
        
        print(f"📄 Detailed analysis saved to: {report_path}")
        return memotion_df
        
    else:
        print("❌ No valid samples were processed!")
        return None

if __name__ == "__main__":
    process_memotion_dataset()