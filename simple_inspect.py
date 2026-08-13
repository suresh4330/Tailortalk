import csv

csv_file = 'byrappa_tejas_31july.csv'

with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames
    print("Columns:", columns)
    row_count = 0
    for row in reader:
        row_count += 1
    print("Number of data rows:", row_count)