import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from matplotlib.colors import ListedColormap

# Add custom font path to Times New Roman
font_path = 'Times New Roman.ttf'  # 替换为字体文件的路径
font_prop = font_manager.FontProperties(fname=font_path)
font_prop_blod = font_manager.FontProperties(fname=font_path, weight='extra bold', size=20)

# Data from the table
# methods = ['TINet', 'Cascade R-CNN', 'RetinaNet', 'FCOS', 'ATSS', 'GFL', 'FCOS w/ RFLA', 'QueryDet', 'QFDet', 'QFDet*', 'COXNet', 'COXNet*', 'CFT', 'HRFuser']
# fps = [13.0, 11.8, 19.4, 19.2, 19.0, 18.7, 13.9, 12.8, 14.2, 5.7, 19.9, 13.9, 10.8, 4.5]
# map50 = [28.30, 31.55, 22.87, 29.89, 36.24, 39.74, 38.20, 37.07, 42.08, 46.72, 45.57, 50.04, 37.32, 33.24]
# flops = [92.80, 76.23, 49.48, 47.21, 48.73, 49.22, 110.62, 121.67, 81.43, 242.82, 51.38, 123.59, 112.02, 54.17]
methods = ['TINet', 'Cascade R-CNN', 'RetinaNet', 'ATSS', 'GFL', 'FCOS w/ RFLA', 'QueryDet', 'QFDet', 'QFDet*', 'COXNet', 'COXNet*', 'CFT']
fps = [13.0, 11.8, 19.4, 19.0, 18.7, 13.9, 12.8, 14.2, 5.7, 17.6, 12.9, 10.8]
map50 = [28.30, 31.55, 22.87, 36.24, 39.74, 38.20, 37.07, 42.08, 46.72, 45.57, 50.04, 22.69]
flops = [92.80, 76.23, 49.48, 48.73, 49.22, 110.62, 121.67, 81.43, 242.82, 51.38, 123.59, 112.02]

# Normalize the Flops for better visualization (for circle size)
norm_flops = [f / max(flops) for f in flops]

# Define custom colors using ListedColormap
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#6a51a3', '#a55194', 
          '#d6616b', '#3182bd']  # Manually chosen colors
custom_cmap = ListedColormap(colors)

# Create the plot with Set3 colormap for lighter colors
plt.figure(figsize=(10, 7))
scatter = plt.scatter(fps, map50, s=[n * 8000 for n in norm_flops], c=np.arange(len(fps)), cmap='Set3', alpha=0.7, edgecolors='black', linewidth=0.5)

# Add black dots in the center of the circles
plt.scatter(fps, map50, color='black', s=10)  # Black center points

# Annotate some key methods in red
highlighted_methods = ['COXNet', 'COXNet*']
# for i, method in enumerate(methods):
#     if method in highlighted_methods:
#         plt.text(fps[i] + 0.1, map50[i], method, fontsize=16, color='red', fontproperties=font_prop_blod, fontweight='bold')
#     else:
#         plt.text(fps[i] + 0.1, map50[i], method, fontsize=14, fontproperties=font_prop)

# Labels and title with custom font
plt.xlabel('Speed (fps)', fontproperties=font_prop, fontsize=20)
plt.ylabel('RGBTDronePerson mAP$_{50}$ (%)', fontproperties=font_prop, fontsize=20)
# plt.title('Performance Comparison: FPS vs mAP$_{50}$ (%) on RGBTDronePerson Dataset', fontproperties=font_prop, fontsize=16)

# Adjust font size of the tick labels on the axes
plt.tick_params(axis='both', which='major', labelsize=12)  # Adjusted fontsize for the tick labels

# Remove color bar
# plt.colorbar(scatter, label='Method Index (for visualization)')

# Add grid and layout adjustments
plt.grid(True)
plt.tight_layout()

# Save the figure to 1.jpg
plt.savefig('1.jpg', format='jpg', dpi=300)

# Optionally, to display the plot (can be removed if not needed)
# plt.show()