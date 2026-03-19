import matplotlib.pyplot as plt
import numpy as np

# Light direction from above (0°) to below (180°)
light_angle = np.linspace(0, 180, 200)

# 1. Shadows only (binary step)
shadows = np.where(light_angle < 90, 1, 0)

# 2. Shading only (smooth sigmoid)
shading = 1 / (1 + np.exp((light_angle - 90)/15))

# 3. Shadow-biased smooth curve
shadow_bias = 1 / (1 + np.exp((light_angle - 95)/8))  # steeper, slightly shifted toward shadow flip

plt.figure(figsize=(6,4))
plt.plot(light_angle, shadows, color='green', linewidth=2)
plt.plot(light_angle, shading, color='blue', linewidth=2)
plt.plot(light_angle, shadow_bias, color='red', linewidth=2)

plt.title('Convexity Perception vs. Light Direction')
plt.xlim(0, 180)
plt.ylim(0, 1)

plt.show()