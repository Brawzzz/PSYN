import colorsys

def generate_colors(n):

    colors = []

    for i in range(n):
        
        hue = i / n
        saturation = 1.0
        lightness = 1.0
        
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, lightness)

        colors.append((int(b*255), int(g*255), int(r*255)))
        
    return colors