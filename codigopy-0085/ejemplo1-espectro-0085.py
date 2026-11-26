import os
import matplotlib.pyplot as plt
from skimage import io, color, util
print("Ricardo Lopez 0085")
carpeta_0 = 'funciones-frecuencia-0085'
os.makedirs(carpeta_0, exist_ok=True)

imagen_original = io.imread('tejido.png')

if imagen_original.ndim == 3:
    if imagen_original.shape[2] == 4:
        imagen_original = color.rgba2rgb(imagen_original)
    imagen_gris = color.rgb2gray(imagen_original)
else:
    imagen_gris = util.img_as_float(imagen_original)

plt.imsave(os.path.join(carpeta_0, 'imagen_original_grises.png'), imagen_gris, cmap='gray')
print("Listo: imagen_original_grises.png guardada.")
print("Ricardo Lopez 0085")