import math
import numpy as np
import matplotlib.pyplot as plt

# totally random value (for plotting)
SIZE = 10

# plot area size
RESOLUTION = 100


class FastVectorLookup:
    def __init__(self, table_size=65536):
        self.table_size = table_size
        self.vectors = self._generate_unit_vectors(table_size)
    
    def _generate_unit_vectors(self, n):
        # Generate 2D unit vectors
        vectors = np.zeros((n, 2))
        theta = 2 * np.pi * np.random.rand(n)
        vectors[:, 0] = np.cos(theta)
        vectors[:, 1] = np.sin(theta)
        return vectors
    
    def get_unit_vectors(self, x, y):
        # Vectorized hash function for 2D
        h = (x * 73856093) ^ (y * 19349663)
        h = ((h >> 16) ^ h) * 0x45d9f3b
        h = ((h >> 16) ^ h) * 0x45d9f3b
        h = (h >> 16) ^ h
        
        indices = h % self.table_size
        return self.vectors[indices]


fast_lookup = FastVectorLookup()


def get_unit_vector(x, y):
    return fast_lookup.get_unit_vector(x, y)


def perlin(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    original_shape = x.shape
    
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    cell_x = np.floor(x_flat).astype(int)
    cell_y = np.floor(y_flat).astype(int)
    
    dx = x_flat - cell_x
    dy = y_flat - cell_y
    
    g00 = fast_lookup.get_unit_vectors(cell_x,     cell_y)
    g10 = fast_lookup.get_unit_vectors(cell_x + 1, cell_y)
    g01 = fast_lookup.get_unit_vectors(cell_x,     cell_y + 1)
    g11 = fast_lookup.get_unit_vectors(cell_x + 1, cell_y + 1)
    
    dp00 = g00[:, 0] * dx + g00[:, 1] * dy
    dp10 = g10[:, 0] * (dx - 1) + g10[:, 1] * dy
    dp01 = g01[:, 0] * dx + g01[:, 1] * (dy - 1)
    dp11 = g11[:, 0] * (dx - 1) + g11[:, 1] * (dy - 1)
    
    sx = dx * dx * dx * (dx * (dx * 6 - 15) + 10)
    sy = dy * dy * dy * (dy * (dy * 6 - 15) + 10)
    
    x00 = dp00 + sx * (dp10 - dp00)
    x10 = dp01 + sx * (dp11 - dp01)
    
    result = x00 + sy * (x10 - x00)

    return result.reshape(original_shape)


# add octaves
def octave_perlin(x, y, octaves, persistence):
    x = np.asarray(x)
    y = np.asarray(y)

    total = np.zeros_like(x, dtype=float)
    #total = 0.0
    frequency = 1.0
    amplitude = 1.0
    max_value = 0.0
    
    for _ in range(octaves):
        total += perlin(x * frequency, y * frequency) * amplitude

        max_value += amplitude
        amplitude *= persistence
        # times lacunarity value
        frequency *= 2

    normalized = total / max_value
    return np.clip((normalized + 1.0) / 2.0, 0.0, 1.0)


def plot_noise():
    x = np.linspace(0, SIZE, RESOLUTION)
    y = np.linspace(0, SIZE, RESOLUTION)
    x_coords, y_coords = np.meshgrid(x, y)

    noise = octave_perlin(x_coords, y_coords, octaves=4, persistence=0.5)
    
    # Normalize for display
    noise_min, noise_max = noise.min(), noise.max()
    if noise_max > noise_min:
        noise = (noise - noise_min) / (noise_max - noise_min)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.imshow(noise, cmap="gray", extent=[0, SIZE, 0, SIZE], origin='lower')
    plt.colorbar()
    plt.title('2D Perlin Noise')
    plt.show()
    

if __name__ == "__main__":
    plot_noise()