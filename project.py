import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from PIL import Image, ImageTk
import tkinter as tk

global loaded_image

def convolution(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculer la taille de la sortie
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    # Initialiser la matrice de sortie
    output = np.zeros((output_height, output_width), dtype=np.float32)

    # Effectuer la convolution
    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(image[i:i + kernel_height, j:j + kernel_width] * kernel)

    return output

def sobel_operator(image):
    # Définition des filtres Sobel pour la détection des bords en x et y
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Appliquer la convolution pour détecter les bords en x et y
    gradient_x = convolution(image, sobel_x)
    gradient_y = convolution(image, sobel_y)

    # Calculer le gradient total
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    return gradient_magnitude


def load_image():
    global loaded_image
    
    image_path = filedialog.askopenfilename()  
    if image_path:  
        loaded_image = Image.open(image_path)
        loaded_image.thumbnail((700, 700))  
        
        photo = ImageTk.PhotoImage(loaded_image)
        label.config(image=photo)
        label.image = photo
          # Ajouter le La


# def load_image():
#     image_path = filedialog.askopenfilename()  
#     if image_path:  
#         loaded_image = Image.open(image_path)
#         loaded_image.thumbnail((300, 700))  
        
#         photo = ImageTk.PhotoImage(loaded_image)
#         label.config(image=photo)
#         label.image = photo
        
#         # apply_sobel_button.config(command=lambda: apply_sobel(image_path))
#         # apply_prewitt_button.config(command=lambda: apply_prewitt(image_path))
#         # apply_robinson_button.config(command=lambda: apply_robinson(image_path))
#         # apply_laplacian_button.config(command=lambda: apply_laplacian(image_path))
#         # apply_segmentation_button.config(command=lambda: apply_segmentation(image_path))

def display_images_side_by_side(result_sobel_custom, result_sobel_opencv):
    # Convertir les images résultantes en format PIL
    pil_sobel_custom = Image.fromarray(result_sobel_custom).convert("RGB")
    pil_sobel_opencv = Image.fromarray(result_sobel_opencv).convert("RGB")

    # Créer une nouvelle image pour afficher les images côte à côte
    width = pil_sobel_custom.width + pil_sobel_opencv.width
    height = max(pil_sobel_custom.height, pil_sobel_opencv.height)
    composite_image = Image.new('RGB', (width, height))

    # Coller les images l'une à côté de l'autre dans l'image composite
    composite_image.paste(pil_sobel_custom, (0, 0))
    composite_image.paste(pil_sobel_opencv, (pil_sobel_custom.width, 0))

    # Convertir l'image composite en PhotoImage pour l'afficher dans Tkinter
    composite_photo = ImageTk.PhotoImage(composite_image)
    
    # Afficher l'image composite dans une étiquette Tkinter
    composite_label = tk.Label(root, image=composite_photo)
    composite_label.image = composite_photo
    composite_label.pack()  


def apply_sobel():
    global loaded_image
    if loaded_image:
        # Convertir l'image PIL en tableau NumPy
        image_np = np.array(loaded_image)
        # Convertir en niveaux de gris si nécessaire
        if len(image_np.shape) > 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Appliquer l'opérateur Sobel (custom)
        result_sobel_custom = sobel_operator(image_np)

        # Appliquer l'opérateur Sobel prédéfini de OpenCV
        result_sobel_opencv_x = cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=3)
        result_sobel_opencv_y = cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=3)
        result_sobel_opencv = np.sqrt(result_sobel_opencv_x ** 2 + result_sobel_opencv_y ** 2) 
        

        # Afficher les images résultantes avec Matplotlib
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(result_sobel_custom, cmap='gray')
        plt.title('Sobel Operator (Custom)')

        plt.subplot(1, 2, 2)
        plt.imshow(result_sobel_opencv, cmap='gray')
        plt.title('Sobel Operator (OpenCV)')

        plt.show()
def apply_prewitt():
    global loaded_image
    if loaded_image:
        # Convertir l'image PIL en tableau NumPy
        image_np = np.array(loaded_image)
        # Convertir en niveaux de gris si nécessaire
        if len(image_np.shape) > 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        c = 1  # Paramètre de normalisation pour Prewitt
        prewitt_x = np.array([[1, 0, -1], [c, 0, -c], [1, 0, -1]]) / (2 * c + 1)
        prewitt_y = np.array([[-1, -c, -1], [0, 0, 0], [1, c, 1]]) / (2 * c + 1)

        prewitt_x_filtered = convolution(image_np, prewitt_x)
        prewitt_y_filtered = convolution(image_np, prewitt_y)
        prewitt_magnitude = np.sqrt(prewitt_x_filtered ** 2 + prewitt_y_filtered ** 2)

        # Afficher l'image résultante dans Tkinter
        # display_image_in_tkinter(prewitt_magnitude, 'Prewitt')  # À implémenter

        # Afficher l'image résultante avec Matplotlib (optionnel)
        plt.imshow(prewitt_magnitude, cmap='gray')
        plt.title('Prewitt')
        plt.show()

def robinson_operator(image, kernel):
    # Appliquer la convolution pour détecter les bords
    gradient = convolution(image, kernel)

    return gradient

def apply_robinson():
    global loaded_image
    if loaded_image:
        # Convertir l'image PIL en tableau NumPy
        image_np = np.array(loaded_image)
        # Convertir en niveaux de gris si nécessaire
        if len(image_np.shape) > 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Définir le noyau Robinson 0
        robinson_0 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        # Appliquer l'opérateur Robinson 0
        result_robinson = convolution(image_np, robinson_0)

        # Afficher l'image résultante dans Tkinter
        # display_image_in_tkinter(result_robinson, 'Robinson Operator (Direction 0)')  # À implémenter

        # Afficher l'image résultante avec Matplotlib (optionnel)
        plt.figure(figsize=(5, 5))
        plt.imshow(result_robinson, cmap='gray')
        plt.title('Robinson Operator (Direction 0)')
        plt.show()




def laplacian_operator(image, kernel):
    # Appliquer la convolution pour détecter les bords
    gradient = convolution(image, kernel)

    return gradient





def image_segmentation(image, threshold_value):
    # Convertir l'image en niveaux de gris si nécessaire
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Appliquer le seuillage pour segmenter l'image
    _, segmented_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Trouver les contours dans l'image segmentée
    contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Créer une copie de l'image originale pour dessiner les contours
    image_with_contours = image.copy()
    
    # Dessiner les contours sur l'image copiée
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
    
    # Retourner l'image avec les contours et le masque binaire
    return image_with_contours, segmented_image

def apply_segmentation():
    global loaded_image
    if loaded_image:
        # Convertir l'image PIL en tableau NumPy
        image_np = np.array(loaded_image)
        threshold_value = 200
        segmented_image_rgb, segmented_image = image_segmentation(image_np, threshold_value)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(segmented_image, cmap='gray')
        plt.title('Segmented Image')
        plt.subplot(1, 2, 2)
        plt.imshow(segmented_image_rgb)
        plt.title('Segmentation element similaire')
        plt.show()


def apply_laplacian():
    global loaded_image
    if loaded_image:
        # Convertir l'image PIL en tableau NumPy
        image_np = np.array(loaded_image)
        # Convertir en niveaux de gris si nécessaire
        if len(image_np.shape) > 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Définir le noyau Laplacien
        laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

        # Appliquer l'opérateur Laplacien
        result_laplacian = convolution(image_np, laplacian)

        # Afficher l'image résultante dans Tkinter
        # display_image_in_tkinter(result_laplacian, 'Laplacian Operator')  # À implémenter

        # Afficher l'image résultante avec Matplotlib (optionnel)
        plt.figure(figsize=(5, 5))
        plt.imshow(result_laplacian, cmap='gray')
        plt.title('Laplacian Operator')
        plt.show()

# morphologie opera
def erosion(image):
    # Convertir l'image en niveaux de gris si nécessaire
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image

    # Élément structurant pour l'érosion
    kernel = np.ones((3, 3), np.uint8)

    # Appliquer l'érosion sur l'image en niveaux de gris
    eroded_image = cv2.erode(gray_image, kernel, iterations=1)

    return eroded_image



# def erosion(image):
#     # Convertir l'image en niveaux de gris si nécessaire
#     if len(image.shape) > 2:
#         gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     else:
#         gray_image = image

#     # Élément structurant pour l'érosion
#     trans = np.ones((3, 3), dtype=np.uint8)

#     # Dimensions de l'image
#     l, c = gray_image.shape

#     # Résultat de l'érosion
#     result = np.zeros_like(gray_image)

#     for i in range(1, l-1):
#         for j in range(1, c-1):
#             if np.all(gray_image[i-1:i+2, j-1:j+2] & trans == trans):
#                 result[i, j] = 255
#             else:
#                 result[i, j] = 0

#     return result

def apply_erosion():
    global loaded_image
    if loaded_image:
        # Convertir l'image chargée en tableau NumPy
        image_np = np.array(loaded_image)
        
        # Appliquer l'érosion sur l'image en niveaux de gris
        eroded_image = erosion(image_np)

        # Afficher l'image érodée avec Matplotlib
        plt.imshow(eroded_image, cmap='gray')
        plt.title('Erosion')
        plt.show()

def dilation(image):
    # Convertir l'image en niveaux de gris si nécessaire
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image

    # Élément structurant pour la dilatation
    trans = np.ones((3, 3), dtype=np.uint8)

    # Dimensions de l'image
    l, c = gray_image.shape

    # Résultat de la dilatation
    result = np.zeros_like(gray_image)

    for i in range(1, l-1):
        for j in range(1, c-1):
            if np.any(gray_image[i-1:i+2, j-1:j+2] & trans == trans):
                result[i, j] = 255
            else:
                result[i, j] = 0

    return result

def apply_dilation():
    global loaded_image
    if loaded_image:
        # Convertir l'image chargée en tableau NumPy
        image_np = np.array(loaded_image)
        
        # Appliquer la dilatation sur l'image
        dilated_image = dilation(image_np)

        # Afficher l'image dilatée avec Matplotlib
        plt.imshow(dilated_image, cmap='gray')
        plt.title('Dilation')
        plt.show() 
    

def opening(image):
    # Convertir l'image en niveaux de gris si nécessaire
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image

    # Élément structurant pour l'ouverture
    trans = np.ones((3, 3), dtype=np.uint8)

    # Dimensions de l'image
    l, c = gray_image.shape

    # Résultat de l'ouverture
    result = np.zeros_like(gray_image)

    # Appliquer l'érosion suivie de la dilatation
    for i in range(1, l-1):
        for j in range(1, c-1):
            eroded = np.all(gray_image[i-1:i+2, j-1:j+2] & trans == trans)
            if eroded:
                result[i, j] = 255

    for i in range(1, l-1):
        for j in range(1, c-1):
            dilated = np.any(result[i-1:i+2, j-1:j+2] & trans == trans)
            if dilated:
                result[i, j] = 255

    return result

def apply_opening():
    global loaded_image
    if loaded_image: 
        kernel = np.ones((3, 3), dtype=np.uint8)
        # Convertir l'image chargée en tableau NumPy
        image_np = np.array(loaded_image)
        opened_image_cv = cv2.morphologyEx(image_np, cv2.MORPH_OPEN,kernel)
       
        # Appliquer l'ouverture sur l'image
        opened_image = opening(image_np)

      # Affichage côte à côte
        # plt.subplot(1, 2, 1)
        # plt.imshow(opened_image, cmap='gray')
        # plt.title('Opening')
        # plt.axis('off')  # Pour supprimer les axes si besoin

        plt.subplot(1, 2, 2)
        plt.imshow(opened_image_cv, cmap='gray')
        plt.title('Opening ')
        plt.axis('off')  # Pour supprimer les axes si besoin

        plt.show()


def closing(image):
    # Convertir l'image en niveaux de gris si nécessaire
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image

    # Élément structurant pour la fermeture
    trans = np.ones((3, 3), dtype=np.uint8)

    # Dimensions de l'image
    l, c = gray_image.shape

    # Résultat de la fermeture
    result = np.zeros_like(gray_image)

    # Appliquer la dilatation suivie de l'érosion
    for i in range(1, l-1):
        for j in range(1, c-1):
            dilated = np.any(gray_image[i-1:i+2, j-1:j+2] & trans == trans)
            if dilated:
                result[i, j] = 255

    for i in range(1, l-1):
        for j in range(1, c-1):
            eroded = np.all(result[i-1:i+2, j-1:j+2] & trans == trans)
            if eroded:
                result[i, j] = 255

    return result

def apply_closing():
    global loaded_image
    if loaded_image:
        # Convertir l'image chargée en tableau NumPy
        image_np = np.array(loaded_image)
        
        # Appliquer la fermeture sur l'image
        closed_image = closing(image_np)

        # Afficher l'image fermée avec Matplotlib
        plt.imshow(closed_image, cmap='gray')
        plt.title('Closing')
        plt.show()
  
    
    
    
    
    
    
    
root = tk.Tk()
root.title("HOME")


# Charger l'image
image_path = "bg.jpg"  # Remplacez par le chemin de votre image
img = Image.open(image_path)
# img = img.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.ANTIALIAS)
img = img.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.ANTIALIAS if "ANTIALIAS" in dir(Image) else Image.BILINEAR)


# Convertir l'image pour l'afficher dans Tkinter
background_image = ImageTk.PhotoImage(img)

# Créer un label pour afficher l'image en tant que fond
background_label = tk.Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)



load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack()

# Frame pour organiser les boutons sur une ligne
# buttons_frame = tk.Frame(root)
# buttons_frame.pack()

# apply_sobel_button = tk.Button(buttons_frame, text="Apply Sobel Filter", command=lambda: None)
# apply_sobel_button.pack(side=tk.LEFT, padx=5, pady=5)

# apply_prewitt_button = tk.Button(buttons_frame, text="Apply Prewitt Filter", command=lambda: None)
# apply_prewitt_button.pack(side=tk.LEFT, padx=5, pady=5)

# apply_robinson_button = tk.Button(buttons_frame, text="Apply Robinson Filter", command=lambda: None)
# apply_robinson_button.pack(side=tk.LEFT, padx=5, pady=5)

# apply_laplacian_button = tk.Button(buttons_frame, text="Apply Laplacian Filter", command=lambda: None)
# apply_laplacian_button.pack(side=tk.LEFT, padx=5, pady=5)

# apply_segmentation_button = tk.Button(buttons_frame, text="Apply Segmentation", command=lambda: None)
# apply_segmentation_button.pack(side=tk.LEFT, padx=0, pady=5)

def fixed_thresholding_custom(image, threshold):
    if len(image.shape) > 2:
        # Convertir en niveaux de gris si l'image est en couleur
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Taille de l'image
    height, width = image.shape

    # Créer une nouvelle image pour stocker le résultat
    thresholded_image = np.zeros((height, width), dtype=np.uint8)

    # Parcourir tous les pixels de l'image
    for i in range(height):
        for j in range(width):
            # Appliquer le seuillage
            if image[i, j] > threshold:
                thresholded_image[i, j] = 255  # Pixel blanc si supérieur au seuil
            else:
                thresholded_image[i, j] = 0  # Pixel noir sinon

    return thresholded_image


def apply_threshold():
    global loaded_image
    if loaded_image:
        # Convertir l'image chargée en tableau NumPy
        image_np = np.array(loaded_image)
        
        # Vérifier si l'image est en niveaux de gris ou en couleur
        if len(image_np.shape) > 2:  # S'il y a plus de 2 dimensions, c'est une image couleur
            # Convertir l'image couleur en niveaux de gris
            image_np_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            # Appliquer le seuillage binaire
            threshold = 127  # Seuil (à personnaliser)
            thresholded_image = fixed_thresholding_custom(image_np_gray, threshold)
        else:  # Si l'image est en niveaux de gris
            # Appliquer le seuillage binaire directement
            threshold = 127  # Seuil (à personnaliser)
            thresholded_image = fixed_thresholding_custom(image_np, threshold)

        # Afficher l'image seuillée avec Matplotlib
        plt.imshow(thresholded_image, cmap='gray')
        plt.title('Seuillage binaire')
        plt.show()

def average_filter(image):
    rows, cols = image.shape[:2]  # Récupérer les dimensions de l'image
    filtered_image = np.zeros((rows - 2, cols - 2), dtype=np.uint8)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Calculer la moyenne des pixels environnants
            average = int(np.mean([
                image[i - 1, j - 1], image[i - 1, j], image[i - 1, j + 1],
                image[i, j - 1], image[i, j], image[i, j + 1],
                image[i + 1, j - 1], image[i + 1, j], image[i + 1, j + 1]
            ]))  # Convertir en entier

            filtered_image[i - 1, j - 1] = average

    return filtered_image




def apply_average_filter():
    global loaded_image
    if loaded_image:
        # Convertir l'image chargée en tableau NumPy
        image_np = np.array(loaded_image)
        
        # Appliquer le filtre moyenneur 3x3
        filtered_image = average_filter(image_np)

        # Affichage de l'image filtrée avec Matplotlib
        plt.imshow(filtered_image, cmap='gray')
        plt.title('Filtre Moyenneur 3x3')
        plt.show()




def average_filter_5(image):
    rows, cols = image.shape[:2]  # Récupérer les dimensions de l'image
    filtered_image = np.zeros((rows - 4, cols - 4), dtype=np.uint8)

    for i in range(2, rows - 2):
        for j in range(2, cols - 2):
            # Calculer la moyenne des pixels environnants dans un voisinage 5x5
            average = int(np.mean([
                image[i - 2, j - 2], image[i - 2, j - 1], image[i - 2, j], image[i - 2, j + 1], image[i - 2, j + 2],
                image[i - 1, j - 2], image[i - 1, j - 1], image[i - 1, j], image[i - 1, j + 1], image[i - 1, j + 2],
                image[i, j - 2], image[i, j - 1], image[i, j], image[i, j + 1], image[i, j + 2],
                image[i + 1, j - 2], image[i + 1, j - 1], image[i + 1, j], image[i + 1, j + 1], image[i + 1, j + 2],
                image[i + 2, j - 2], image[i + 2, j - 1], image[i + 2, j], image[i + 2, j + 1], image[i + 2, j + 2]
            ]))  # Convertir en entier

            filtered_image[i - 2, j - 2] = average

    return filtered_image

def apply_average_filter_5():
    global loaded_image
    if loaded_image:
        # Convertir l'image chargée en tableau NumPy
        image_np = np.array(loaded_image)

        # Appliquer un filtre moyenneur 5x5
        filtered_image = average_filter_5(image_np)

        # Affichage de l'image filtrée avec Matplotlib
        plt.imshow(filtered_image, cmap='gray')
        plt.title('Filtre Moyenneur 5x5')
        plt.show()


def apply_gamma_correction():
    gamma = 0.5
    global loaded_image
    if loaded_image:
        # Convertir l'image chargée en tableau NumPy
        image_np = np.array(loaded_image)
        print('the shape is ', image_np.shape)
        
        # Normaliser les valeurs de pixel pour l'image
        normalized_image = image_np / 255.0

        # Appliquer la correction gamma
        if len(image_np.shape) == 2:  # Si l'image est en niveaux de gris (2D)
            gamma_corrected = (normalized_image ** gamma) * 255
        else:  # Si l'image est en couleur (3D)
            gamma_corrected = np.zeros_like(normalized_image)
            for i in range(normalized_image.shape[0]):
                for j in range(normalized_image.shape[1]):
                    for k in range(normalized_image.shape[2]):
                        gamma_corrected[i, j, k] = (normalized_image[i, j, k] ** gamma) * 255

        # Affichage de l'image avec Matplotlib
        plt.imshow(gamma_corrected.astype(np.uint8), cmap='gray')
        plt.title(f'Correction Gamma (gamma={gamma})')
        plt.show()
        
        
        
# def erosion(image):
#     # Seuillage de l'image initiale
#     # _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    
#     # trans = np.ones((3, 3), dtype=np.uint8)  # Élément structurant
#     l, c, _ = image.shape  # Obtenir les dimensions de l'image seuillée

#     result = np.zeros_like(image)  # Crée une image vide de la même taille que l'image seuillée

#     for i in range(1, l-1):
#         for j in range(1, c-1):
#             if np.all(image[i-1:i+2, j-1:j+2] & image == image):
#                 result[i, j] = 1
#             else:
#                 result[i, j] = 0

#     return result

#  morpholigical 



label = tk.Label(root)
label.pack()


# Création de la barre de menu
menubar = tk.Menu(root)
root.config(menu=menubar)

# Création du menu déroulant
filter_menu = tk.Menu(menubar, tearoff=0)
filter_menu = tk.Menu(menubar, tearoff=0, fg="purple", bg="pink")  # Exemple de couleurs pour le texte et le fond

menubar.add_cascade(label="Contours", menu=filter_menu)

# Ajout des options dans le menu déroulant
filter_menu.add_command(label="Apply Sobel ", command=apply_sobel)
filter_menu.add_command(label="Apply Prewitt ", command=apply_prewitt) 
filter_menu.add_command(label="Apply Robinson ", command=apply_robinson)
filter_menu.add_command(label="Apply Laplacian ", command=apply_laplacian)
# filter_menu.add_command(label="Apply Segmentation", command=apply_segmentation)

    


# Menu pour la binarisation

bina = tk.Menu(menubar, tearoff=0)
bina = tk.Menu(menubar, tearoff=0,  fg="purple", bg="pink")  # Exemple de couleurs pour le texte et le fond

menubar.add_cascade(label="Binarisation", menu=bina)
bina.add_command(label="Seuillage binaire", command=apply_threshold)

filter_menu = tk.Menu(menubar, tearoff=0)
filter_menu = tk.Menu(menubar, tearoff=0,  fg="purple", bg="pink")  # Exemple de couleurs pour le texte et le fond

menubar.add_cascade(label="Filtrage", menu=filter_menu)

filter_menu.add_command(label="average filtre 3 X 3  ", command=apply_average_filter)
filter_menu.add_command(label="average filtre 5 X 5  ", command=apply_average_filter_5)
filter_menu.add_command(label="gamma filtre 0.5", command=apply_gamma_correction)


# Ajout à la barre de menu
erosion_menu = tk.Menu(menubar, tearoff=0)
erosion_menu = tk.Menu(menubar, tearoff=0,  fg="purple", bg="pink")  # Exemple de couleurs pour le texte et le fond

menubar.add_cascade(label="Morpho operation", menu=erosion_menu)
erosion_menu.add_command(label="Apply Erosion", command=apply_erosion)
erosion_menu.add_command(label="Apply  dilation", command=apply_dilation)
erosion_menu.add_command(label="opening ", command=apply_opening)
erosion_menu.add_command(label="closing", command=apply_closing)




seg_mnu = tk.Menu(menubar, tearoff=0)
seg_mnu = tk.Menu(menubar, tearoff=0,  fg="purple", bg="pink")  # Exemple de couleurs pour le texte et le fond

menubar.add_cascade(label="segmentation", menu=seg_mnu)
seg_mnu.add_command(label="segmentation", command=apply_segmentation)


from tkinter import messagebox

def afficher_aide():
    texte_aide = (
        "Contour :\n Les opérateurs de détection de contours comme Sobel, Prewitt, Roberts et Laplacien sont utilisés pour détecter les transitions d'intensité significatives dans une image. Sobel, Prewitt et Roberts détectent les contours dans différentes directions, tandis que Laplacien met en évidence les zones de changements rapides dans les niveaux de gris de l'image.\n\n" +
        
        "Binarisation :\nLa binarisation convertit une image en une image binaire en seuillant les niveaux de gris. Ce processus simplifie l'image en distinguant les objets d'intérêt du fond. En fixant un seuil, les pixels au-dessus deviennent blancs et en dessous deviennent noirs.\n\n" +
        
        "Filtrage :\nLe filtrage en traitement d'image utilise des filtres pour des opérations comme la suppression du bruit ou la modification de l'apparence. Le filtre moyen réduit le bruit en remplaçant la valeur de chaque pixel par la moyenne de ses voisins. Le filtre gamma modifie la luminosité ou le contraste de l'image.\n\n" +
        
        "Morphologie :\nLes opérations morphologiques modifient la forme des objets dans une image. L'érosion diminue la taille des objets en enlevant les pixels des bords extérieurs. La dilatation augmente la taille des objets en ajoutant des pixels aux bords extérieurs. L'ouverture et la fermeture sont des combinaisons d'érosion et de dilatation utilisées pour traiter les détails des objets.\n\n" +
        
        "Segmentation :\nLa segmentation divise une image en régions pour simplifier l'analyse. Cela peut inclure la détection de contours pour séparer les objets, la binarisation pour distinguer les régions d'intérêt, ou d'autres techniques pour regrouper les régions similaires en fonction de certaines caractéristiques."
    )
    
    messagebox.showinfo("Aide - Concepts de traitement d'image", texte_aide)

menubar.add_cascade(label="Help !!",command=afficher_aide )  

root.mainloop()
