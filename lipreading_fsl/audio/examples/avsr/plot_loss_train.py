import os
import pandas as pd
import matplotlib.pyplot as plt

# Define hardcoded paths
input_path = "/home/frb6002/Documents/experiments/offline_model_lr_3e-4/lightning_logs/version_2/metrics.csv"
output_dir = "/home/frb6002/Documents/experiments/offline_model_lr_3e-4/lightning_logs/version_2/plots"

# Ensure output directory exists
if not os.path.exists(output_dir):
    print(f"Output directory does not exist: {output_dir}")
    exit(1)

# Load the CSV file
if not os.path.exists(input_path):
    print(f"Error: Input file not found: {input_path}")
    exit(1)

df = pd.read_csv(input_path)

# Check if necessary columns exist
required_columns = ['step', 'Losses/train_loss_step', 'Losses/val_loss_step']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"Error: Missing columns in CSV: {missing_columns}")
    exit(1)

# Define parameters
num_plots = 1       # Number of plots
x_range = 24000     # Steps per plot
y_max = 500         # Cap y-axis at 500

# Convert `step` column to numeric to avoid errors
df['step'] = pd.to_numeric(df['step'], errors='coerce')

# Ensure step values are unique by averaging duplicates
df = df.groupby('step', as_index=False).mean()

# Handle missing values properly before interpolation
df_filtered = df[['step', 'Losses/train_loss_step', 'Losses/val_loss_step']].set_index('step')

# Interpolate missing values and clip at y_max
df_interpolated = df_filtered.interpolate()
df_interpolated_clipped = df_interpolated.clip(upper=y_max)

# Identify outliers (values above y_max)
train_outliers = df_interpolated[df_interpolated['Losses/train_loss_step'] > y_max]
val_outliers = df_interpolated[df_interpolated['Losses/val_loss_step'] > y_max]

# Find the actual min and max step values
min_step = df_interpolated_clipped.index.min()
max_step = df_interpolated_clipped.index.max()

# Generate and save each plot
for i in range(num_plots):
    start_x = max(i * x_range, min_step)  # Ensure we don't start before actual data
    end_x = min(start_x + x_range, max_step)

    # Skip if the step range is beyond actual data
    if start_x > max_step:
        print(f"Skipping plot {i+1} (no data in steps {start_x} to {end_x})")
        continue

    # Find closest step indices using absolute difference
    df_reset = df_interpolated_clipped.reset_index()
    start_idx = (df_reset['step'] - start_x).abs().idxmin()
    end_idx = (df_reset['step'] - end_x).abs().idxmin()

    # Extract the closest available step range using iloc
    train_subset = df_reset.iloc[start_idx:end_idx]
    val_subset = df_reset.iloc[start_idx:end_idx]
    train_outliers_subset = train_outliers.reset_index().iloc[start_idx:end_idx]
    val_outliers_subset = val_outliers.reset_index().iloc[start_idx:end_idx]

    # Skip empty plots
    if train_subset.empty and val_subset.empty:
        print(f"Skipping empty plot {i+1} (steps {start_x} to {end_x})")
        continue

    # Select every 10th step for plotting
    train_subset_downsampled = train_subset.iloc[::10]  # Every 10th step
    val_subset_downsampled = val_subset.iloc[::10]  # Every 10th step

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot downsampled training loss (solid line)
    if not train_subset_downsampled.empty:
        plt.plot(train_subset_downsampled['step'], train_subset_downsampled['Losses/train_loss_step'],
                 color='blue', linestyle='-', linewidth=2, label='Train Loss')

    # Plot downsampled validation loss (solid line, steps multiplied by 10)
    if not val_subset_downsampled.empty:
        plt.plot(val_subset_downsampled['step'] * 10, val_subset_downsampled['Losses/val_loss_step'],
                 color='purple', linestyle='-', linewidth=2, label='Validation Loss (Step x100)')

    # Plot outliers as dots only
    if not train_outliers_subset.empty:
        plt.scatter(train_outliers_subset['step'], train_outliers_subset['Losses/train_loss_step'],
                    color='blue', marker='o', alpha=0.5, s=20)
    if not val_outliers_subset.empty:
        plt.scatter(val_outliers_subset['step'] * 10, val_outliers_subset['Losses/val_loss_step'],
                    color='purple', marker='o', alpha=0.5, s=20)

    # Adjust x-axis and y-axis limits
    plt.xlim(start_x, end_x)
    plt.ylim(0, y_max)

    # Labels and title
    plt.xlabel('Step')
    plt.ylabel(f'Loss (Capped at {y_max})')
    plt.title(f'Steps {start_x} to {end_x} (Every 10th Step)')
    plt.legend()
    plt.grid(True)

    # Save the figure
    output_path = os.path.join(output_dir, f'loss_plot_{i+1}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")

print(f"\nAll plots saved in {output_dir}")
