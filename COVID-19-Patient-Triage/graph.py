import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Load Data ---
DATA_FILE = 'covid_symptoms_severity_prediction.csv'
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Data file '{DATA_FILE}' not found. Ensure it is in the same directory.")
    exit()

# --- 2. Create Target Variable ---
# Define severity: 1 if hospitalized OR icu_admission OR mortality is 1, else 0.
df['severity_flag_numeric'] = np.where(
    (df['hospitalized'] == 1) | 
    (df['icu_admission'] == 1) | 
    (df['mortality'] == 1), 
    1, 
    0
)

# Convert numeric flag to descriptive labels for plotting
df['severity_outcome'] = df['severity_flag_numeric'].apply(
    lambda x: 'Severe' if x == 1 else 'Non-Severe'
)

# --- 3. Generate Violin Plot (Age Distribution by Severity) ---
plt.figure(figsize=(9, 6))
sns.violinplot(
    x='severity_outcome', 
    y='age', 
    data=df, 
    palette={'Severe': '#e63946', 'Non-Severe': '#457b9d'} # Custom colors for clarity
)

plt.title('Age Distribution by COVID-19 Severity Outcome', fontsize=16)
plt.xlabel('Severity Outcome', fontsize=12)
plt.ylabel('Patient Age', fontsize=12)
plt.yticks(np.arange(0, 101, 10)) # Set y-axis ticks in steps of 10
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Save the plot
plt.savefig('age_severity_violin_plot.png')
plt.close()

print("✅ Plot 'age_severity_violin_plot.png' generated successfully.")