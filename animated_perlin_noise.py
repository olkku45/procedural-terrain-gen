import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# totally random value (for plotting),
# the bigger, the more pixels, and laggier
SIZE = 2

# plot area size
RESOLUTION = 100

# FPS = 1000 / TIME_BETWEEN_FRAMES
# currently FPS is about 30
TIME_BETWEEN_FRAMES = 33


class FastVectorLookup:
    def __init__(self, table_size=65536):
        self.table_size = table_size
        self.vectors = self._generate_unit_vectors(table_size)
    
    def _generate_unit_vectors(self, n):
        vectors = np.zeros((n, 3))
        z = 2.0 * np.random.rand(n) - 1.0
        theta = 2 * np.pi * np.random.rand(n)
        r = np.sqrt(1 - z * z)
        vectors[:, 0] = r * np.cos(theta)
        vectors[:, 1] = r * np.sin(theta)
        vectors[:, 2] = z
        return vectors
    
    def get_unit_vectors(self, x, y, z):
        # Vectorized hash function
        h = (x * 73856093) ^ (y * 19349663) ^ (z * 83492791)
        h = ((h >> 16) ^ h) * 0x45d9f3b
        h = ((h >> 16) ^ h) * 0x45d9f3b
        h = (h >> 16) ^ h
        
        indices = h % self.table_size
        return self.vectors[indices]


fast_lookup = FastVectorLookup()

# calculate perlin noise values, all operations in this function
# except gradient vector calculations
def perlin(x, y, z):
    x = np.asarray(x)
    y = np.asarray(y) 
    z = np.asarray(z)
    original_shape = x.shape
    
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    
    cell_x = np.floor(x_flat).astype(int)
    cell_y = np.floor(y_flat).astype(int)
    cell_z = np.floor(z_flat).astype(int)
    
    dx = x_flat - cell_x
    dy = y_flat - cell_y
    dz = z_flat - cell_z
    
    g000 = fast_lookup.get_unit_vectors(cell_x,     cell_y,     cell_z)
    g100 = fast_lookup.get_unit_vectors(cell_x + 1, cell_y,     cell_z)
    g010 = fast_lookup.get_unit_vectors(cell_x,     cell_y + 1, cell_z)
    g110 = fast_lookup.get_unit_vectors(cell_x + 1, cell_y + 1, cell_z)
    g001 = fast_lookup.get_unit_vectors(cell_x,     cell_y,     cell_z + 1)
    g101 = fast_lookup.get_unit_vectors(cell_x + 1, cell_y,     cell_z + 1)
    g011 = fast_lookup.get_unit_vectors(cell_x,     cell_y + 1, cell_z + 1)
    g111 = fast_lookup.get_unit_vectors(cell_x + 1, cell_y + 1, cell_z + 1)
    
    dp000 = g000[:, 0] * dx + g000[:, 1] * dy + g000[:, 2] * dz
    dp100 = g100[:, 0] * (dx - 1) + g100[:, 1] * dy + g100[:, 2] * dz
    dp010 = g010[:, 0] * dx + g010[:, 1] * (dy - 1) + g010[:, 2] * dz
    dp110 = g110[:, 0] * (dx - 1) + g110[:, 1] * (dy - 1) + g110[:, 2] * dz
    dp001 = g001[:, 0] * dx + g001[:, 1] * dy + g001[:, 2] * (dz - 1)
    dp101 = g101[:, 0] * (dx - 1) + g101[:, 1] * dy + g101[:, 2] * (dz - 1)
    dp011 = g011[:, 0] * dx + g011[:, 1] * (dy - 1) + g011[:, 2] * (dz - 1)
    dp111 = g111[:, 0] * (dx - 1) + g111[:, 1] * (dy - 1) + g111[:, 2] * (dz - 1)
    
    sx = dx * dx * dx * (dx * (dx * 6 - 15) + 10)
    sy = dy * dy * dy * (dy * (dy * 6 - 15) + 10)
    sz = dz * dz * dz * (dz * (dz * 6 - 15) + 10)
    
    x00 = dp000 + sx * (dp100 - dp000)
    x10 = dp010 + sx * (dp110 - dp010)
    x01 = dp001 + sx * (dp101 - dp001)
    x11 = dp011 + sx * (dp111 - dp011)
    
    y0 = x00 + sy * (x10 - x00)
    y1 = x01 + sy * (x11 - x01)
    
    result = y0 + sz * (y1 - y0)
    
    return result.reshape(original_shape)

# add octaves
def octave_perlin(x, y, t, octaves, persistence):
    x = np.asarray(x)
    y = np.asarray(y)

    total = np.zeros_like(x, dtype=float)
    # defaults: (1.0, 1.0, 0.0)
    frequency = 1.0
    amplitude = 1.0
    max_value = 0.0
    
    for _ in range(octaves):
        total += perlin(x * frequency, y * frequency, t * frequency) * amplitude

        max_value += amplitude
        amplitude *= persistence
        # times lacunarity value
        frequency *= 2

    normalized = total / max_value
    
    return np.clip((normalized + 1.0) / 2.0, 0.0, 1.0)


def update(frame):
    t = frame * 0.1

    noise = octave_perlin(x_coords, y_coords, t, 4, 0.5)

    # normalize noise values for plot
    noise_min, noise_max = noise.min(), noise.max()
    if noise_max > noise_min:
        noise = (noise - noise_min) / (noise_max - noise_min)

    im.set_array(noise)
    return [im]


x = np.linspace(0, SIZE, RESOLUTION)
y = np.linspace(0, SIZE, RESOLUTION)
x_coords, y_coords = np.meshgrid(x, y)

fig, ax = plt.subplots(figsize=(8, 6))
# look up matplotlib cmaps for different color maps
im = ax.imshow(np.zeros((RESOLUTION, RESOLUTION)), cmap="hsv", vmin=0, vmax=1)
plt.colorbar(im, ax=ax)


def plot_noise():
    anim = FuncAnimation(fig, update, frames=100, interval=TIME_BETWEEN_FRAMES, blit=True)
    plt.show()
    

if __name__ == "__main__":
    plot_noise()