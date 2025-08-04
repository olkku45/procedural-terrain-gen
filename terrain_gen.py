from perlin_noise import octave_perlin
import pygame
import numpy as np
import cProfile

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
pygame.display.set_caption("Terrain map generator")

font = pygame.font.Font("freesansbold.ttf", 32)
small_font = pygame.font.Font("freesansbold.ttf", 24)

# initialize surfaces which will be drawn
color_surfaces = {}
for terrain_type, color in COLORS.items():
    surface = pygame.Surface((TILE_SIZE, TILE_SIZE))
    surface.fill(color)
    color_surfaces[terrain_type] = surface


def get_noise_values(width, height, tile_size):
    x_coords = np.arange(0, width, tile_size)
    y_coords = np.arange(0, height, tile_size)
    X, Y = np.meshgrid(x_coords, y_coords)
    x_flat = X.flatten() * SCALE
    y_flat = Y.flatten() * SCALE
    
    # get 'raw' noise values, not properly scaled between 0-1
    raw_noise_values = octave_perlin(x_flat, y_flat, octaves=4, persistence=0.5)

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
            elif noise_val < 0.65:
                terrain_type = "grass"
            elif noise_val < 0.85:
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
    
    drawn = False
    terrain_surface = None

    start_time = pygame.time.get_ticks()

    noise_values = get_noise_values(WIDTH, HEIGHT, TILE_SIZE)
    terrain_surface = generate_terrain_surface(WIDTH, HEIGHT, TILE_SIZE, noise_values)
            
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        if not drawn:
            generation_time = pygame.time.get_ticks() - start_time
            print(f"Terrain map generation took: {generation_time}ms")

            screen.blit(terrain_surface, (0, 0))
            pygame.display.flip()
            drawn = True
        
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    #cProfile.run('main()')
    main()

# TILE_SIZE = 10:
# (optimizations in this file and perlin_noise)
# without optimizations: 20s
# with optimizations, without progress bar: 3s
# with optimizations, with progress bar: 5-7s
# actually optimized: 10-30 ms lolll