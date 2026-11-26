import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util
print("Ricardo Lopez 0085")
carpeta_3 = 'ejemplo3-sobel-y-0085'
os.makedirs(carpeta_3, exist_ok=True)

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

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

fft_sobel_y = np.fft.fftshift(np.fft.fft2(sobel_y, s=(filas, columnas)))
fft_filtrada_y = fft_centrada * fft_sobel_y
espectro_filtrado_y = np.log(1 + np.abs(fft_filtrada_y))
bordes_y = np.real(np.fft.ifft2(np.fft.ifftshift(fft_filtrada_y)))

plt.imsave(os.path.join(carpeta_3, 'espectro_filtrado_sobel_y.png'), espectro_filtrado_y, cmap='inferno')
plt.imsave(os.path.join(carpeta_3, 'bordes_sobel_y.png'), np.abs(bordes_y), cmap='gray')
print("Listo: resultados de Sobel Y guardados.")
print("Ricardo Lopez 0085")