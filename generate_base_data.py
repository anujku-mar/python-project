import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

# 1. Load the FULL diabetes dataset
print("Loading real diabetes data...")
# Make sure this file is in the same folder as this script
file_name = 'diabetes_binary_health_indicators_BRFSS2015.csv'
real_data = pd.read_csv(file_name)

print(f"Dataset loaded successfully! Total rows: {len(real_data)}")

# Note: We do NOT need to drop any columns here because there are no Names or Dates!

# 2. Detect the Metadata
# SDV needs to know which columns are numbers and categories
print("Detecting metadata...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)

# 3. Initialize the Gaussian Copula Synthesizer
print("Initializing the Gaussian Copula model...")
synthesizer = GaussianCopulaSynthesizer(metadata)

# 4. Train the Model on the real data
print("Training the model (Note: Because this file has 250,000+ rows, this may take a few minutes)...")
synthesizer.fit(real_data)

# 5. Generate the Synthetic Data
# Generating 50,000 rows for your group to sample from
num_rows_to_generate = 50000 
print(f"Generating {num_rows_to_generate} synthetic rows...")
synthetic_data = synthesizer.sample(num_rows=num_rows_to_generate)

# 6. Save the output to a new CSV file
output_filename = 'synthetic_diabetes_master_pool.csv'
synthetic_data.to_csv(output_filename, index=False)
print(f"Success! Synthetic diabetes data saved to {output_filename}")