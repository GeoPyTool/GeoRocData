import matplotlib.colors as mcolors
import numpy as np

categories = ["INTRAPLATE VOLCANICS", "RIFT VOLCANICS", "OCEAN ISLAND", "OCEANIC PLATEAU", 
              "SUBMARINE RIDGE", "CONVERGENT MARGIN", "CONTINENTAL FLOOD BASALT", 
              "ARCHEAN CRATON (INCLUDING GREENSTONE BELTS)", "null"]

# Generate evenly spaced hues
hues = np.linspace(0, 1, len(categories) + 1)[:-1]

# Set saturation and lightness
saturation = 0.5
lightness = 0.5

color_dict = {}
for category, hue in zip(categories, hues):
    # Convert HSL to RGB
    rgb = mcolors.hsv_to_rgb([hue, saturation, lightness])
    # Convert RGB to RGBA
    rgba = list(rgb) + [1.0]  # Alpha is set to 1.0
    color_dict[category] = rgba

print(color_dict)