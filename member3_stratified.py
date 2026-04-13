import pandas as pd

print("1. Loading datasets...")
# Load the REAL data to calculate the exact real-world percentages
real_data = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

# Load the SYNTHETIC data to actually draw the new sample from
synthetic_pool = pd.read_csv('synthetic_diabetes_master_pool.csv')

print("2. Calculating the true real-world proportions...")
# Get the exact percentage breakdown of the target column
real_proportions = real_data['Diabetes_binary'].value_counts(normalize=True)

print("\n--- Target Real-World Proportions ---")
print(real_proportions)
print("-------------------------------------")

print("\n3. Performing Stratified Sampling...")
# We want a final dataset of exactly 5,000 rows
TARGET_TOTAL_ROWS = 5000
stratified_blocks = []

# Loop through each category (0.0 for Healthy, 1.0 for Diabetic)
for condition_value, proportion in real_proportions.items():
    
    # Calculate exactly how many rows are needed to match the real percentage
    rows_needed = int(TARGET_TOTAL_ROWS * proportion)
    print(f" -> Pulling {rows_needed} rows for Diabetes_binary = {condition_value}")
    
    # Filter the synthetic pool for this specific condition
    synthetic_subset = synthetic_pool[synthetic_pool['Diabetes_binary'] == condition_value]
    
    # Randomly draw the required number of rows from this subset
    # random_state=42 ensures the exact same rows are drawn if you run the script again
    sampled_block = synthetic_subset.sample(n=rows_needed, random_state=42)
    
    # Store this sampled block
    stratified_blocks.append(sampled_block)

print("\n4. Stitching the final dataset together...")
# Combine the healthy block and the diabetic block into one final dataframe
final_stratified_sample = pd.concat(stratified_blocks)

# Save the final dataset to a new CSV file
output_filename = 'member3_stratified_sample.csv'
final_stratified_sample.to_csv(output_filename, index=False)

print(f"Success! Stratified dataset saved to: {output_filename}")
print(f"Total rows in final sample: {len(final_stratified_sample)}")