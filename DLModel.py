import tarfile
import csv
import pandas as pd

def tar_to_csv(tar_file, csv_name):
    # Find the CSV file within the tar.gz file
    csv_file = None
    for file in tar_file.getmembers():
        if file.name.endswith('.csv'):
            csv_file = file
            break

    # Read the CSV file if found and save it as a separate CSV file with integers
    if csv_file:
        # Extract the CSV file from the tar.gz file
        extracted_csv = tar_file.extractfile(csv_file)

        # Read the CSV data
        csv_reader = csv.reader((line.decode() for line in extracted_csv))  # Decode bytes to strings

        # Create a new CSV file to save the converted data as integers
        output_csv_filename = csv_name
        with open(output_csv_filename, 'w', newline='') as output_csv_file:
            csv_writer = csv.writer(output_csv_file)

            # Convert and write each row from the extracted CSV file to the output CSV file
            for row in csv_reader:
                csv_writer.writerow(row)

        # Close the extracted and output CSV files
    extracted_csv.close()
    output_csv_file.close()
    print(f"CSV file extracted and saved as {output_csv_filename}")
    tar_file.close()

# # convert tar file to csv
# newyork_data = tarfile.open('1minute_data_newyork.tar.gz', 'r:gz')
# tar_to_csv(newyork_data, 'newyork_csv')
