import pandas as pd
import os
import sys

def filter_csv(file_path):
    print(f"Processing: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
        if 'daySaleStrategy' not in df.columns:
            print(f"Error: Column 'daySaleStrategy' not found in {file_path}")
            print(f"Columns found: {df.columns.tolist()}")
            return
        
        # Filter: keep rows where daySaleStrategy <= 5
        # Ensure the column is numeric just in case
        df['daySaleStrategy'] = pd.to_numeric(df['daySaleStrategy'], errors='coerce')
        filtered_df = df[df['daySaleStrategy'] <= 5]
        
        # Generate output filename
        dir_name, file_name = os.path.split(file_path)
        name, ext = os.path.splitext(file_name)
        output_file = os.path.join(dir_name, f"{name}_le5d{ext}")
        
        filtered_df.to_csv(output_file, index=False)
        print(f"Success! Filtered data saved to: {output_file}")
        print(f"Original rows: {len(df)}")
        print(f"Filtered rows: {len(filtered_df)}")
        print(f"Removed rows: {len(df) - len(filtered_df)}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Default file if no argument provided
    default_file = '/Users/frank/work/code/lh/lh/pc/csv/yield_hits_browser_createTime_wrgt55_20260130_161801.csv'
    
    target_file = default_file
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    
    filter_csv(target_file)
