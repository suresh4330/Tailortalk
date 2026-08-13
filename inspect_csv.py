import csv
import os

csv_file = 'byrappa_tejas_31july.csv'

# Check if file exists
if not os.path.exists(csv_file):
    print(f"File {csv_file} not found.")
    exit(1)

# Initialize counters and data structures
row_count = 0
columns = []
image_urls = []
missing_values = {col: 0 for col in []}
duplicate_urls = set()
seen_urls = set()

# Read the CSV
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    # Get the column names from the reader's fieldnames
    columns = reader.fieldnames
    print(f"Column names: {columns}")
    print(f"Number of columns: {len(columns)}")

    # Initialize missing values dictionary for each column
    missing_values = {col: 0 for col in columns}

    for row in reader:
        row_count += 1
        # Check for missing values in each column
        for col in columns:
            if not row[col] or row[col].strip() == '':
                missing_values[col] += 1

        # Collect image URLs
        url = row['image_url']
        image_urls.append(url)
        if url in seen_urls:
            duplicate_urls.add(url)
        else:
            seen_urls.add(url)

print(f"\nNumber of data rows: {row_count}")
print(f"\nMissing values per column:")
for col, count in missing_values.items():
    print(f"  {col}: {count}")

print(f"\nTotal image URLs collected: {len(image_urls)}")
print(f"Unique image URLs: {len(seen_urls)}")
print(f"Duplicate image URLs: {len(duplicate_urls)}")
if duplicate_urls:
    print("List of duplicate URLs (first 10):")
    for i, url in enumerate(list(duplicate_urls)[:10]):
        print(f"  {url}")
    if len(duplicate_urls) > 10:
        print(f"  ... and {len(duplicate_urls) - 10} more")

# Additionally, let's check the first few rows to see the data
print("\nFirst 2 rows:")
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 2:
            break
        print(f"Row {i+1}:")
        for col in columns:
            print(f"  {col}: {row[col]}")