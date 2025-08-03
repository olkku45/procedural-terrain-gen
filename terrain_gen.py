from perlin_noise import octave_perlin
import pygame
import numpy as np

WIDTH, HEIGHT = 1280, 720
# if tile_size < 4, it may be
# difficult to exit out during
# the loading screen :) at
# least for me, it will differ
# greatly based on your hardware
TILE_SIZE = 4
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


def draw_loading_screen(progress, stage_text):
    screen.fill((0, 0, 0))
    
    loading_text = font.render("Generating Terrain...", True, (255, 255, 255))
    title_rect = loading_text.get_rect()
    title_rect.center = (WIDTH // 2, HEIGHT // 2 - 60)
    screen.blit(loading_text, title_rect)
    
    stage_surface = small_font.render(stage_text, True, (200, 200, 200))
    stage_rect = stage_surface.get_rect()
    stage_rect.center = (WIDTH // 2, HEIGHT // 2 - 20)
    screen.blit(stage_surface, stage_rect)
    
    bar_width = 400
    bar_height = 20
    bar_x = WIDTH // 2 - bar_width // 2
    bar_y = HEIGHT // 2 + 20
    
    pygame.draw.rect(screen, (64, 64, 64), (bar_x, bar_y, bar_width, bar_height))
    
    fill_width = int(bar_width * progress)
    if fill_width > 0:
        pygame.draw.rect(screen, (0, 255, 0), (bar_x, bar_y, fill_width, bar_height))
    
    pygame.draw.rect(screen, (255, 255, 255), (bar_x, bar_y, bar_width, bar_height), 2)
    
    percent_text = small_font.render(f"{int(progress * 100)}%", True, (255, 255, 255))
    percent_rect = percent_text.get_rect()
    percent_rect.center = (WIDTH // 2, HEIGHT // 2 + 60)
    screen.blit(percent_text, percent_rect)
    
    pygame.display.flip()


# we do things in numpy and batches now for performance
def get_noise_values(width, height, tile_size, progress_callback):
    x_coords = np.arange(0, width, tile_size)
    y_coords = np.arange(0, height, tile_size)
    
    X, Y = np.meshgrid(x_coords, y_coords)
    
    x_flat = X.flatten() * SCALE
    y_flat = Y.flatten() * SCALE
    
    raw_noise_values = []
    
    batch_size = 100
    total_points = len(x_flat)

    batches_per_update = max(1, total_points // (batch_size * 20))

    for i in range(0, len(x_flat), batch_size):
        batch_x = x_flat[i:i+batch_size]
        batch_y = y_flat[i:i+batch_size]
        
        for x, y in zip(batch_x, batch_y):
            raw_noise_values.append(octave_perlin(x, y, OCTAVES, PERSISTENCE))

        batch_number = i // batch_size
        if batch_number % batches_per_update == 0 or i + batch_size >= total_points:
            progress = min((i + batch_size) / total_points * 0.6, 0.6)
            progress_callback(progress, "Generating noise values...")
    
    progress_callback(0.6, "Normalizing noise values...")
    
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
def generate_terrain_surface(width, height, tile_size, noise_grid, progress_callback):
    terrain_surface = pygame.Surface((width, height))

    total_tiles = len(noise_grid)
    tiles_processed = 0

    min_update_interval = max(1, min(200, total_tiles // 20))
    
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

            tiles_processed += 1
            progress = 0.6 + (tiles_processed / total_tiles * 0.4)
            
            if tiles_processed % min_update_interval == 0 or tiles_processed == total_tiles:
                progress_callback(progress, "Rendering terrain...")
    
    return terrain_surface

# run program
# optionally measure how long map gen took 
def main():
    clock = pygame.time.Clock()
    run = True
    
    drawn = False
    terrain_surface = None

    start_time = pygame.time.get_ticks()
    
    def update_progress(progress, stage_text):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                nonlocal run
                run = False
        
        if run:
            draw_loading_screen(progress, stage_text)
            clock.tick(FPS)
    
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        if not drawn and run:
            # generate terrain with progress updates
            noise_values = get_noise_values(WIDTH, HEIGHT, TILE_SIZE, update_progress)
            
            if run:  # check if we haven't quit during noise generation
                terrain_surface = generate_terrain_surface(WIDTH, HEIGHT, TILE_SIZE, noise_values, update_progress)
                
                if run:  # check if we haven't quit during terrain generation
                    update_progress(1.0, "Complete!")

                    generation_time = pygame.time.get_ticks() - start_time
                    print(f"Terrain map generation took: {generation_time}ms")
                    
                    pygame.time.wait(500)
                    
                    screen.blit(terrain_surface, (0, 0))
                    pygame.display.flip()
                    drawn = True
        
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()

# TILE_SIZE = 10:
# without optimizations: 20s
# with optimizations, without progress bar: 3s
# with optimizations, with progress bar: 5s