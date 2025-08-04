## Procedural terrain map and animated noise generator, utilizing perlin noise. 

Many optimizations have been made to the original, including vectorization, and 
getting rid of some horrible code that slowed the program to a crawl. With the very first iteration of
the program, generating a terrain map took 20s. Now with the same settings it takes about 20ms, so quite a jump.
Below are previews of what the animated noise and terrain maps may look like (EPILEPSY WARNING):

# Gifs

Wonky and fast looking noise, using 'flag' color map
![gif1](/gifs/perlin_noise_1.gif "gif1")

Noise in heatmap colors, using 'plasma' color map
![gif2](/gifs/perlin_noise_2.gif "gif2")

Greyscale noise, very fast though, using 'gray' color map
![gif3](/gifs/perlin_noise_3.gif "gif3")

Cool looking rainbow noise, using 'hsv' color map
![gif4](/gifs/perlin_noise_4.gif "gif4")

# Images

A terrain map, with mountains, fields, beaches and ponds.
![img1](/img/terrain_map.png "img1")