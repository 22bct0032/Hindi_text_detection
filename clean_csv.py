"""
Script to clean up CSV by removing predicted_words and avg_confidence columns
"""

import csv
import os

def clean_csv(input_csv='outputs/detection_results.csv', output_csv='outputs/detection_results_cleaned.csv'):
    """
    Remove predicted_words, actual_words, and avg_confidence columns from CSV
    Keep only: image_name, predicted_count, actual_count, difference
    """
    print("Cleaning CSV file...")
    
    # Read the original CSV
    with open(input_csv, 'r', encoding='utf-8-sig') as infile:
        reader = csv.DictReader(infile)
        
        # Define new fieldnames (without predicted_words, actual_words, and avg_confidence)
        new_fieldnames = ['image_name', 'predicted_count', 'actual_count', 'difference']
        
        # Read all rows
        rows = []
        for row in reader:
            # Keep only the columns we want
            new_row = {field: row.get(field, '') for field in new_fieldnames}
            rows.append(new_row)
    
    # Write the cleaned CSV
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Cleaned CSV saved to: {output_csv}")
    print(f"✓ Removed columns: predicted_words, actual_words, avg_confidence")
    print(f"✓ Kept columns: {', '.join(new_fieldnames)}")
    print(f"✓ Total rows: {len(rows)}")
    
    # Also overwrite the original file
    with open(input_csv, 'w', newline='', encoding='utf-8-sig') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Original file updated: {input_csv}")

if __name__ == "__main__":
    clean_csv()
