import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util
print("Ricardo Lopez 0085")
carpeta_1 = 'ejemplo1-espectro-0085'
os.makedirs(carpeta_1, exist_ok=True)

imagen_original = io.imread('tejido.png')

if imagen_original.ndim == 3:
    if imagen_original.shape[2] == 4:
        imagen_original = color.rgba2rgb(imagen_original)
    imagen_gris = color.rgb2gray(imagen_original)
else:
    imagen_gris = util.img_as_float(imagen_original)

fft_imagen = np.fft.fft2(imagen_gris)
fft_centrada = np.fft.fftshift(fft_imagen)
espectro_original = np.log(1 + np.abs(fft_centrada))

plt.imsave(os.path.join(carpeta_1, 'espectro_original_fft.png'), espectro_original, cmap='inferno')
print("Listo: espectro_original_fft.png guardado.")
print("Ricardo Lopez 0085")