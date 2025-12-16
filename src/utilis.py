import colorsys

def is_pow_2(n):
    return((n and (n-1)) == 0)

def compute_row_col(n):

    if(n == 2):
        n_row = 2
        n_col = 1
        return (n_row, n_col)
    
    n_row = 2
    n_col = int(n / n_row)

    while(n_col % 2 == 0 and n_col > 4):

        n_row *= 2
        n_col = int(n_col / 2)        

        print(n_row, n_col)

    return(n_row, n_col)

def split_image(img, nb_split=2):

    img_list = []
    
    if(not is_pow_2(nb_split)):

        print("***** WARNING *****")
        print(f"nb_split must be a power of two : nb_split = {nb_split}")
        print("*******************")

        return img_list

    img_w, img_h = img.shape[:2]

    (n_row, n_col) = compute_row_col(nb_split) 

    # for i in range(n_row):

    #     img_list.append(img_i)

    return img_list

def join_image(img_list):

    img = 0

    return img

def generate_colors(n):

    colors = []

    for i in range(n):

        hue = i / n
        saturation = 1.0
        lightness = 1.0
        
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, lightness)

        colors.append((int(b*255), int(g*255), int(r*255)))
        
    return colors


def f_pass(x):
    pass


nb_split = 2

for i in range(8):

    (n_row, n_col) = compute_row_col(nb_split) 
    print(f"nb_split = {nb_split}")
    print(f"n_row = {n_row}")
    print(f"n_col = {n_col}\n")

    nb_split += 2

