from perlin_noise import octave_perlin
import pygame
import numpy as np

WIDTH, HEIGHT = 1280, 720
TILE_SIZE = 10
SCALE = 0.01
FPS = 30

COLORS = {
    "water": (0, 0, 255),
    "sand": (194, 178, 128),
    "grass": (0, 255, 0),
    "rock": (128, 128, 128),
    "snow": (255, 255, 255),
}

OCTAVES = 6
PERSISTENCE = 0.5

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF | pygame.HWSURFACE)

# initialize surfaces which will be drawn
color_surfaces = {}
for terrain_type, color in COLORS.items():
    surface = pygame.Surface((TILE_SIZE, TILE_SIZE))
    surface.fill(color)
    color_surfaces[terrain_type] = surface

# we do things in numpy and batches now for performance
def get_noise_values_vectorized(width, height, tile_size):
    x_coords = np.arange(0, width, tile_size)
    y_coords = np.arange(0, height, tile_size)
    
    X, Y = np.meshgrid(x_coords, y_coords)
    
    x_flat = X.flatten() * SCALE
    y_flat = Y.flatten() * SCALE
    
    raw_noise_values = []
    
    batch_size = 100
    for i in range(0, len(x_flat), batch_size):
        batch_x = x_flat[i:i+batch_size]
        batch_y = y_flat[i:i+batch_size]
        
        for x, y in zip(batch_x, batch_y):
            raw_noise_values.append(octave_perlin(x, y, OCTAVES, PERSISTENCE))
    
    raw_noise_array = np.array(raw_noise_values)
    min_observed = np.min(raw_noise_array)
    max_observed = np.max(raw_noise_array)
    
    if max_observed != min_observed:
        normalized_array = (raw_noise_array - min_observed) / (max_observed - min_observed)
    
    noise_values = {}
    i = 0
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            noise_values[(x, y)] = normalized_array[i]
            i += 1
    
    return noise_values

# generate a blank terrain surface onto which 
# the terrain map is rendered
def generate_terrain_surface(width, height, tile_size, noise_grid):
    terrain_surface = pygame.Surface((width, height))
    
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            noise_val = noise_grid[(x, y)]
            
            if noise_val < 0.20:
                terrain_type = "water"
            elif noise_val < 0.30:
                terrain_type = "sand"
            elif noise_val < 0.60:
                terrain_type = "grass"
            elif noise_val < 0.8:
                terrain_type = "rock"
            else:
                terrain_type = "snow"
            
            terrain_surface.blit(color_surfaces[terrain_type], (x, y))
    
    return terrain_surface

# run program
# optionally measure how long map gen took 
def main():
    clock = pygame.time.Clock()
    run = True

    # start_time = pygame.time.get_ticks()
    
    noise_values = get_noise_values_vectorized(WIDTH, HEIGHT, TILE_SIZE)
    terrain_surface = generate_terrain_surface(WIDTH, HEIGHT, TILE_SIZE, noise_values)
    
    # generation_time = pygame.time.get_ticks() - start_time
    # print(f"Terrain map generation took: {generation_time}ms")
    
    drawn = False
    
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        
        if not drawn:
            screen.blit(terrain_surface, (0, 0))
            pygame.display.flip()
            drawn = True
        
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()

# without optimizations: 20s
# with optimizations: 3s