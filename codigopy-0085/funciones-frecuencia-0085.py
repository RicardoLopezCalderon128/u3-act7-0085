import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util

carpeta_4 = 'ejemplo4-magnitud-gradiente-0085'
os.makedirs(carpeta_4, exist_ok=True)

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

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

fft_sobel_x = np.fft.fftshift(np.fft.fft2(sobel_x, s=(filas, columnas)))
fft_filtrada_x = fft_centrada * fft_sobel_x
bordes_x = np.real(np.fft.ifft2(np.fft.ifftshift(fft_filtrada_x)))

fft_sobel_y = np.fft.fftshift(np.fft.fft2(sobel_y, s=(filas, columnas)))
fft_filtrada_y = fft_centrada * fft_sobel_y
bordes_y = np.real(np.fft.ifft2(np.fft.ifftshift(fft_filtrada_y)))

magnitud_gradiente = np.hypot(bordes_x, bordes_y)
magnitud_gradiente = magnitud_gradiente / np.max(magnitud_gradiente)

plt.imsave(os.path.join(carpeta_4, 'magnitud_gradiente.png'), magnitud_gradiente, cmap='gray')
print("Listo: magnitud_gradiente.png guardada.")
