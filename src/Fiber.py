#----------------------------------------------------------------------------------------------#
#------------------------------------------- IMPORT -------------------------------------------#
#----------------------------------------------------------------------------------------------#
import numpy as np
import cv2 as cv

import tools
import setup as stp


#----------------------------------------------------------------------------------------------#
#------------------------------------------- CLASS --------------------------------------------#
#----------------------------------------------------------------------------------------------#
class Fiber:

    def __init__(self, cnt : list):
        
        self.contour        = cnt
        self.oriented_box   = self.compute_oriented_box()
        self.angle          = self.compute_angle()
        self.position       = self.compute_position()
        self.length         = self.compute_length()
        self.width          = self.compute_width()
        
    #--------------------------------------------------------------------------------#
    def compute_oriented_box(self):

        if(len(self.contour) > 0):
            return(cv.minAreaRect(self.contour))

    #--------------------------------------------------------------------------------#
    def compute_angle(self):
        
        if(self.oriented_box):
            return(fiber_angle(self.oriented_box))

    #--------------------------------------------------------------------------------#
    def compute_position(self):

        if(self.oriented_box):
            return(self.oriented_box[0])
    
    #--------------------------------------------------------------------------------#
    def compute_length(self):

        if(self.oriented_box):
            (w, h) = self.oriented_box[1]
            return(max(w, h))
    
    #--------------------------------------------------------------------------------#
    def compute_width(self):

        if(self.oriented_box):
            (w, h) = self.oriented_box[1]
            return(min(w, h))
        
    #--------------------------------------------------------------------------------#
    def valid_fiber(self) -> bool:

        if(len(self.contour) > stp.CNTRS_LEN_MIN and 
           self.length       >= stp.FIBER_MIN_LEN and 
           self.width        <= stp.FIBER_MAX_WIDTH):
            return True
        
        return False
    
    #--------------------------------------------------------------------------------#
    def draw_fiber(self, img : np.ndarray, reg_angle : float = None) :

        if(tools.img_empty(img)):
            raise ValueError(f"draw_fibers() Fiber.py line 63 : img is empty")
        
        if(reg_angle):
            color = tools.angle_color(reg_angle, config_path=stp.CONFIG_COLOR_PATH)
        else:
            color = tools.angle_color(self.angle, config_path=stp.CONFIG_COLOR_PATH)

        cv.drawContours(image=img, contours=[self.contour], 
                        contourIdx=-1, color=color, 
                        thickness=stp.FIB_THICHNESS, lineType=cv.LINE_AA)

    #--------------------------------------------------------------------------------#
    def print(self):

        print(f"angle           = {self.angle}")
        print(f"position        = {self.position}")
        print(f"length          = {self.length}")
        print(f"oriented_box    = {self.oriented_box}")
        print(f"cnt.len         = {len(self.contour)}")
        print("\n")

#----------------------------------------------------------------------------------------------#
#---------------------------------------- STATIC METHOD ---------------------------------------#
#----------------------------------------------------------------------------------------------#
@staticmethod
def fiber_angle(rect):

    (_, (w, h), angle) = rect
    
    if w > h:
        real_angle = angle 
    else:
        real_angle = angle + 90

    return real_angle % 180

#--------------------------------------------------------------------------------#
@staticmethod
def select_fiber(fibers : list[Fiber]) -> list[Fiber]:

    selected_fibers = []
    for fib in fibers:
        
        if(fib.valid_fiber()):
            selected_fibers.append(fib)
    
    return selected_fibers

#--------------------------------------------------------------------------------#
# @staticmethod
# def sort_fibers(fibers : list[Fiber]) -> list:

#     if not fibers:
#         return

#     sorted_fibers = []
#     fibers_cpy = fibers.copy()

#     while(len(fibers_cpy) > 0) :

#         ref_fib = fibers_cpy.pop(0)
#         ref_ang = ref_fib.angle

#         next_regions = []
#         current_regions = [ref_fib] 
#         for i in range(len(fibers_cpy)):
            
#             current_fib = fibers_cpy[i]
#             current_angle = current_fib.angle

#             diff = abs(ref_ang - current_angle)
#             if diff > stp.DIFF_ANGLE:
#                 diff = stp.MAX_ANGLE - diff

#             if(diff <= stp.DELTA_ANGLE):
#                 current_regions.append(current_fib)
#             else:
#                 next_regions.append(current_fib)

#         if(len(current_regions) > stp.MIN_REGION_SIZE):
#             sorted_fibers.append(current_regions) 
            
#         fibers_cpy = next_regions    
            
#     return sorted_fibers

#--------------------------------------------------------------------------------#
@staticmethod
def sort_fibers(fibers: list['Fiber']) -> list[list['Fiber']]:
    """
    Trie les fibres par orientation en utilisant une approche par histogramme.
    Identifie les pics dominants et regroupe les fibres autour de ces pics.
    """
    if not fibers:
        return []

    # 1. Récupération des angles
    angles = np.array([fib.angle for fib in fibers])
    
    # --- CONFIGURATION (Peut être déplacé dans setup.py) ---
    BIN_SIZE = 1          # Résolution de l'histogramme en degrés
    SIGMA_SMOOTH = 2.0    # Force du lissage pour éviter les faux pics locaux
    MIN_PEAK_HEIGHT = 5   # Nombre min de fibres pour considérer une direction valide
    # ------------------------------------------------------

    # 2. Création de l'histogramme (0 à 180 degrés)
    # On étend un peu la plage [-10, 190] pour gérer proprement les effets de bord (0/180) lors du lissage
    bins = np.arange(-5, 185, BIN_SIZE)
    
    # Astuce pour la circularité : on duplique les fibres proches de 0 et 180 
    # pour peupler les zones tampons (-10..0 et 180..190)
    angles_extended = list(angles)
    for a in angles:
        if a < 10: angles_extended.append(a + 180)
        elif a > 170: angles_extended.append(a - 180)
    
    hist, bin_edges = np.histogram(angles_extended, bins=bins)

    # 3. Lissage de l'histogramme (Gaussian Blur 1D via OpenCV)
    # Convertir en float32 pour le filter2D
    hist_float = hist.astype(np.float32).reshape(1, -1)
    # Création d'un noyau gaussien 1D
    ksize = int(2 * np.ceil(3 * SIGMA_SMOOTH) + 1)
    hist_smooth = cv.GaussianBlur(hist_float, (ksize, 1), SIGMA_SMOOTH)[0]

    # On recadre l'histogramme pour ne garder que la partie [0, 180] réelle
    # L'index 10 correspond à 0 degré (car on a commencé à -10)
    real_hist = hist_smooth[10:190]
    
    # 4. Détection des pics (Maxima locaux)
    # On cherche les indices où la valeur est supérieure aux voisins
    peaks_indices = []
    for i in range(1, len(real_hist) - 1):
        if real_hist[i-1] < real_hist[i] and real_hist[i] > real_hist[i+1]:
            if real_hist[i] > MIN_PEAK_HEIGHT:
                peaks_indices.append(i)
    
    # Gestion spéciale : Pic à 0/180 (Circularité)
    # Si on a un pic tout au début ET tout à la fin, c'est le même pic physique.
    if len(real_hist) > 0:
        # Check bords (approximatif)
        if (real_hist[0] > real_hist[1] and real_hist[0] > MIN_PEAK_HEIGHT) or \
           (real_hist[-1] > real_hist[-2] and real_hist[-1] > MIN_PEAK_HEIGHT):
             # On ajoute 0 (ou 180) si ce n'est pas déjà détecté
             if 0 not in peaks_indices and 179 not in peaks_indices:
                 peaks_indices.append(0)

    # Si aucun pic détecté (cas rare : distribution uniforme ou vide), on renvoie tout ou rien
    if not peaks_indices:
        return [fibers] # Ou [] selon votre logique métier

    # 5. Classification des fibres
    sorted_groups = {idx: [] for idx in peaks_indices}
    
    for fib in fibers:
        a = fib.angle
        
        # Trouver le pic le plus proche (en gérant la circularité 0-180)
        best_peak = -1
        min_dist = float('inf')
        
        for p in peaks_indices:
            # Distance directe
            d1 = abs(a - p)
            # Distance circulaire (ex: dist entre 2 et 178 doit être 4, pas 176)
            d2 = 180 - d1
            dist = min(d1, d2)
            
            if dist < min_dist:
                min_dist = dist
                best_peak = p
        
        # Seuil d'acceptation : stp.DELTA_ANGLE (défini dans setup.py)
        if min_dist <= stp.DELTA_ANGLE:
            sorted_groups[best_peak].append(fib)
            
    # 6. Formatage de sortie (Liste de listes)
    result = []
    for peak in sorted_groups:
        group = sorted_groups[peak]
        if len(group) > stp.MIN_REGION_SIZE: # Filtre de taille minimale
            result.append(group)
            
    return result

#----------------------------------------------------------------------------------------------#
#------------------------------------------- MAIN ---------------------------------------------#
#----------------------------------------------------------------------------------------------#
if __name__ == "__main__":

    cnt = [0, 1.3, 15.3]
    fib = Fiber(cnt)
    fib.print()

