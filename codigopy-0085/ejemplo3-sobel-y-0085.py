import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util
print("Ricardo Lopez 0085")
carpeta_2 = 'ejemplo2-sobel-x-0085'
os.makedirs(carpeta_2, exist_ok=True)

imagen_original = io.imread('tejido.png')

if imagen_original.ndim == 3:
    if imagen_original.shape[2] == 4:
        imagen_original = color.rgba2rgb(imagen_original)
    imagen_gris = color.rgb2gray(imagen_original)
else:
    imagen_gris = util.img_as_float(imagen_original)

filas, columnas = imagen_gris.shape
fft_imagen = np.fft.fft2(imagen_gris)
fft_centrada = np.fft.fftshift(fft_imagen)

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

fft_sobel_x = np.fft.fftshift(np.fft.fft2(sobel_x, s=(filas, columnas)))
fft_filtrada_x = fft_centrada * fft_sobel_x
espectro_filtrado_x = np.log(1 + np.abs(fft_filtrada_x))
bordes_x = np.real(np.fft.ifft2(np.fft.ifftshift(fft_filtrada_x)))

plt.imsave(os.path.join(carpeta_2, 'espectro_filtrado_sobel_x.png'), espectro_filtrado_x, cmap='inferno')
plt.imsave(os.path.join(carpeta_2, 'bordes_sobel_x.png'), np.abs(bordes_x), cmap='gray')
print("Listo: resultados de Sobel X guardados.")
print("Ricardo Lopez 0085")