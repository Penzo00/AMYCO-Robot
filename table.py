import matplotlib.pyplot as plt

# Data for the points
x_values = [38, 45, 106, 1394, 1736, 1822]
y_values = [82.63, 74, 92.6, 54.2, 80, 97.62]
labels = [
    "Kang et al. 2018",
    "Preechasuk et al. 2019",
    "Kiss et al. 2021",
    "Sulc et al. 2020",
    "Fungus Identification of fungi (mobile app)",
    "A-MYCO prototype 2024",
]

# Normalize x-values for equal spacing
x_positions = range(len(x_values))

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(x_positions, y_values, color=["green", "blue", "red", "orange", "black", "purple"], s=100, label="Methods")

# Annotate each point with its label
for i, label in enumerate(labels):
    plt.text(x_positions[i], y_values[i] + 1, label, fontsize=10, ha='center')

# Set x-axis and y-axis labels
plt.xticks(ticks=x_positions, labels=x_values)
plt.yticks(fontsize=10)
plt.xlabel("Number of species to classify", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)

# Add a grid for better readability
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Add a title
plt.title("Comparison of Methods for Species Classification", fontsize=14)

# Save the plot
output_plot_path = "species_classification_comparison.png"
plt.savefig(output_plot_path)
plt.show()

output_plot_path