import pandas as pd
import os

# Define the folder structure
base_folder = "splits"
subfolders = ["train", "test", "val"]
file_names = {
    "train": "text_train.csv",
    "test": "text_test.csv", 
    "val": "text_val.csv"
}

# List to store all dataframes
dataframes = []

for folder in subfolders:
    file_path = os.path.join(base_folder, folder, file_names[folder])
    
    # Check if file exists
    if os.path.exists(file_path):
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Prune columns - keep only 'text' and 'label'
        columns_to_keep = [col for col in ['text', 'label'] if col in df.columns]
        df_pruned = df[columns_to_keep]
        
        # Add source column to track which split it came from
        df_pruned['source'] = folder
        
        dataframes.append(df_pruned)
        print(f"Processed {file_path}: {len(df_pruned)} rows")
    else:
        print(f"Warning: File not found - {file_path}")

# Combine all dataframes
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Display info about the combined dataset
    print(f"\nCombined dataset shape: {combined_df.shape}")
    print(f"\nRows from each source:")
    print(combined_df['source'].value_counts())
    
    # Save the combined dataframe to CSV
    combined_df.to_csv('combined_text_data.csv', index=False)
    print(f"\n✅ Combined CSV saved as 'combined_text_data.csv'")
    
    # Display first few rows
    print(f"\nFirst few rows of combined data:")
    print(combined_df.head())
    
else:
    print("No data was processed. Please check your file paths.")