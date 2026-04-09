import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

# 1. Load the original dataset
print("Loading real healthcare data...")
real_data = pd.read_csv('healthcare_dataset.csv')

# 2. Drop PII (Personally Identifiable Information) 
# These columns are not needed for statistical sampling and slow down the model
columns_to_drop = ['Name', 'Doctor', 'Hospital']
real_data_cleaned = real_data.drop(columns=columns_to_drop)

# 3. Detect the Metadata
# SDV needs to know which columns are numbers, dates, and categories
print("Detecting metadata...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data_cleaned)

# 4. Initialize the Gaussian Copula Synthesizer
print("Initializing the Gaussian Copula model...")
synthesizer = GaussianCopulaSynthesizer(metadata)

# 5. Train the Model on the real data
print("Training the model (this may take a few minutes depending on your CPU)...")
synthesizer.fit(real_data_cleaned)

# 6. Generate the Synthetic Data
# Generating a large pool so your 3 group members have plenty of data to sample from
num_rows_to_generate = 50000 
print(f"Generating {num_rows_to_generate} synthetic rows...")
synthetic_data = synthesizer.sample(num_rows=num_rows_to_generate)

# 7. Save the output to a new CSV file
output_filename = 'synthetic_master_pool.csv'
synthetic_data.to_csv(output_filename, index=False)
print(f"Success! Synthetic data saved to {output_filename}")