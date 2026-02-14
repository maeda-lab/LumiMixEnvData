import numpy as np
import matplotlib.pyplot as plt
# Times New Romanフォントの設定
plt.rcParams['font.family'] = 'Times New Roman'

# Parameters
x_p = -3                  # Initial x-coordinate of point P
d_p_initial = 5           # Initial depth of point P
v_P = -1                  # Velocity in the d-direction (v_P < 0)
d_e = -3                  # Depth of the eye (fixed)

# Time parameters
time = np.linspace(0, 5, 100)  # Time array (0 to 5 seconds, 100 points)
# d_p = d_p_initial + v_P * time  # Depth of point P as a function of time

d_p = d_p_initial 
x_p = x_p + v_P * time 

# Calculate x (position on the screen) and v_x (velocity on the screen)
x_screen = x_p * (-d_e) / (d_p - d_e)
v_screen = x_p * (d_e * v_P) / (d_p - d_e)**2

# Create the figure
fig, ax = plt.subplots(figsize=(8, 4))

# Plot time vs x (screen position)
ax.plot(x_screen, time, label=r"Screen Position ($x$)", color='blue', lw=2)
ax.text(-1.8, 2, "Position", fontsize=14, color='blue', horizontalalignment='center')

# Plot time vs v_x (screen velocity)
ax.plot(v_screen, time, label=r"Screen Velocity ($v_x$)", color='red', lw=2)
ax.text(-0.5, 2, "Velocity", fontsize=14, color='red', horizontalalignment='center')

# Add horizontal line at t = 5s
ax.axhline(y=5, color='black', linestyle='--', linewidth=1)
ax.text(-2.1, 4.9, "On Screen", fontsize=14, color='black', horizontalalignment='center')

# Customize axis labels and title
# ax.set_xlabel("Value (Position/Velocity)", fontsize=14)
ax.set_ylabel("Time (s)", fontsize=14, labelpad=0)
ax.set_title("Screen Position and Velocity Over Time", fontsize=16)

# Invert y-axis for time to progress downwards
ax.invert_yaxis()

# Remove grid
ax.grid(False)

# Remove grid and frame
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)

# Display both x-axis and y-axis
ax.spines['left'].set_position(('data', 0))  # y-axis at x = 0
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_position(('data', 0))  # x-axis at y = 0
ax.spines['bottom'].set_visible(False)

# Customize ticks
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(axis='both', direction='in')

# Add downward arrow for the y-axis
arrow_properties_y = dict(facecolor='black', edgecolor='black', arrowstyle='-|>', lw=1.5)
ax.annotate("", xy=(0, 5.2), xytext=(0, -0.2), arrowprops=arrow_properties_y)

# Add rightward arrow for the x-axis
arrow_properties_x = dict(facecolor='black', edgecolor='black', arrowstyle='-|>', lw=1.5)
ax.annotate("", xy=(0.00, 0), xytext=(-3.1, 0), arrowprops=arrow_properties_x)

# Add legend in LaTeX format
ax.legend(fontsize=12, loc='center left', frameon=False)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
