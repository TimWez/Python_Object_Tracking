import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.rc('text', usetex=True)     # Enable LaTeX font rendering
plt.rc('font', family='serif')

# Colors for the plot
col_lines = '#2F4F4F'  # Dark Slate Gray
col_graph = '#008080'   # Teal
col_shade = '#ADD8E6'   # Light Blue

shade_areas = True  # Set to True to shade the areas
hide_y_axis = True  # Set to True to hide the y-axis, False to show it

data = [0.245, 0.254, 0.221, 0.235, 0.238,
        0.237, 0.235, 0.229, 0.195, 0.228,
        0.224, 0.232, 0.229, 0.242, 0.257,
        0.216, 0.214, 0.202, 0.201, 0.237]

mean = np.mean(data)
std_dev = np.std(data)
print(mean)
print(std_dev)

# Step 3: Generate x-values for the normal distribution curve
x_values = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 1000)

# Step 4: Calculate the normal distribution values (PDF)
pdf_values = norm.pdf(x_values, mean, std_dev)

label = r'$f_{\mu ,\sigma}(x) = \frac{{1}}{{\sigma \sqrt{{2\pi}}}} e^{-\frac{{(x - \mu)^2}}{{2\sigma^2}}} \quad \mu \approx {mean} \quad \sigma \approx {std_dev}$'


plt.figure(figsize=(10, 4))
plt.plot(x_values, pdf_values, color=col_graph, linewidth=4, label=label)
# Step 6: Add shaded areas for standard deviations
if shade_areas:
    plt.fill_between(x_values, 0, pdf_values, where=(x_values >= mean - std_dev) & (x_values <= mean + std_dev),
                     color=col_shade, alpha=0.4)  # Base alpha
    plt.fill_between(x_values, 0, pdf_values, where=(x_values >= mean - 2 * std_dev) & (x_values <= mean + 2 * std_dev),
                     color=col_shade, alpha=0.2)  # Decreased alpha
    plt.fill_between(x_values, 0, pdf_values, where=(x_values >= mean - 3 * std_dev) & (x_values <= mean + 3 * std_dev),
                     color=col_shade, alpha=0.1)  # Further decreased alpha

# Draw dashed vertical lines at ±1σ, ±2σ, ±3σ, and mean
for i in range(-3, 4):
    x_line = mean + i * std_dev
    y_line = norm.pdf(x_line, mean, std_dev)
    plt.axvline(x_line, color=col_lines, linestyle='--', linewidth=1.5,
                ymax=(y_line / max(pdf_values)) * (0.95 if abs(i) <= 1 else (0.90 if abs(i) == 2 else 0.75)))

# Step 8: Draw the x-axis at y=0
plt.axhline(0, color='black', linewidth=1)

# Step 9: Labeling the plot
#plt.xlabel(r'Data Values', fontsize=14)
plt.ylabel(r'Density', fontsize=14)
plt.legend()

tick_values = [mean - 3 * std_dev, mean - 2 * std_dev, mean - 1 * std_dev, mean,
               mean + 1 * std_dev, mean + 2 * std_dev, mean + 3 * std_dev]
tick_labels = [rf'{mean - 3 * std_dev:.3f}',
               rf'{mean - 2 * std_dev:.3f}',
               rf'{mean - 1 * std_dev:.3f}',
               rf'{mean:.3f}',
               rf'{mean + 1 * std_dev:.3f}',
               rf'{mean + 2 * std_dev:.3f}',
               rf'{mean + 3 * std_dev:.3f}']


# Rotate the x-ticks for better visibility
plt.xticks(tick_values, tick_labels, rotation=45, ha='right')

# Annotate standard deviation lines
for i in range(-3, 4):
    x_line = mean + i * std_dev
    y_line = norm.pdf(x_line, mean, std_dev)

    if i > 0:
        delta = 1
        sign = '+'
    elif i < 0:
        delta = -1
        sign = '-'

    if i == 0:
        plt.text(x_line, y_line * 1.1, r'$\mu$', fontsize=12, color='black', ha='center')
    elif abs(i) == 1:
        plt.text(x_line + (0.0065 * delta), y_line * 1.1, rf'$\mu {sign} 1\sigma$', fontsize=12, color='black', ha='center')
    elif abs(i) == 2:
        plt.text(x_line + (0.0055 * delta), y_line * 1.2, rf'$\mu {sign} 2\sigma$', fontsize=12, color='black', ha='center')
    elif abs(i) == 3:
        plt.text(x_line + (0.005 * delta), y_line * 3.5, rf'$\mu {sign} 3\sigma$', fontsize=12, color='black', ha='center')


# Step 7: Add a red dot in the middle of each shaded area
for i in range(-3, 4):
    if i != 0:  # Skip the mean itself
        if i > 0:
            x_dot = (mean + i * std_dev ) #+ std_dev/2
            y_dot = norm.pdf(x_dot, mean, std_dev)/1.5
            #plt.plot(x_dot - std_dev/2, y_dot, 'ro')  # Red dot
            if i == 1:
                plt.text(x_dot - std_dev/2, y_dot, rf'$34\%$', fontsize=14, color='black', ha='center')
            if i == 2:
                plt.text(x_dot - std_dev/2, y_dot, rf'$13.5\%$', fontsize=12, color='black', ha='center')
            if i == 3:
                plt.text(x_dot - std_dev / 1.25, y_dot/0.5, rf'$2.5\%$', fontsize=10, color='black', ha='center')

        if i < 0:
            x_dot = (mean + i * std_dev ) #+ std_dev/2
            y_dot = norm.pdf(x_dot, mean, std_dev)/1.5
            #plt.plot(x_dot + std_dev/2, y_dot, 'ro')  # Red dot
            if i == -1:
                plt.text(x_dot + std_dev / 2, y_dot, rf'$34\%$', fontsize=14, color='black', ha='center')
            if i == -2:
                plt.text(x_dot + std_dev / 2, y_dot, rf'$13.5\%$', fontsize=12, color='black', ha='center')
            if i == -3:
                plt.text(x_dot + std_dev / 1.25, y_dot/0.5, rf'$2.5\%$', fontsize=10, color='black', ha='center')

# Adjust y-limits to ensure x-axis is clearly at y=0
plt.ylim(0, plt.ylim()[1])

# Hide the top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Step 10: Control variable to hide the y-axis
if hide_y_axis:
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_position(('data', 0))

# Ensure that the plot layout is adjusted
plt.tight_layout()
graph_file = f'Normal_I.png'
plt.savefig(graph_file, dpi=1000)    # Save the graph
# Show the plot
plt.show()


