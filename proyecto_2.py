import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os

# ----------------------------
# Paso 1: Cargar metadatos y seleccionar un archivo
# ----------------------------
df = pd.read_excel("Diagnostics.xlsx")

# Seleccionamos una señal del ritmo 'SR' por ejemplo
archivo_objetivo = df[df["Rhythm"] == "SR"].iloc[0]["FileName"]
ruta_archivo = os.path.join("ECGDataDenoised", archivo_objetivo + ".csv")

# ----------------------------
# Paso 2: Cargar la señal ECG del archivo
# ----------------------------
ecg_data = pd.read_csv(ruta_archivo)
ecg_signal = ecg_data["ECG"].values
fs = 500  # Frecuencia de muestreo usada en el proyecto

# ----------------------------
# Paso 3: Filtrado de la señal ECG (bandpass entre 0.5 y 40 Hz)
# ----------------------------
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

ecg_filtrada = bandpass_filter(ecg_signal, 0.5, 40, fs)

# ----------------------------
# Paso 4: Aplicar FFT y obtener espectro de potencia
# ----------------------------
n = len(ecg_filtrada)
freqs = np.fft.rfftfreq(n, d=1/fs)
fft_vals = np.fft.rfft(ecg_filtrada)
fft_power = np.abs(fft_vals) ** 2

# ----------------------------
# Paso 5: Extraer características espectrales
# ----------------------------
f_dominante = freqs[np.argmax(fft_power)]
energia_total = np.sum(fft_power)
banda_util = freqs[fft_power > 0.05 * np.max(fft_power)]

print(f"✅ Frecuencia dominante: {f_dominante:.2f} Hz")
print(f"✅ Energía total espectral: {energia_total:.2f}")
print(f"✅ Banda útil: {banda_util[0]:.2f} Hz – {banda_util[-1]:.2f} Hz")

# ----------------------------
# Paso 6: Graficar señal y espectro
# ----------------------------
tiempo = np.linspace(0, n/fs, n)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(tiempo, ecg_filtrada)
plt.title("ECG filtrado")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")

plt.subplot(1, 2, 2)
plt.plot(freqs, fft_power)
plt.title("Espectro de Potencia (DFT)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Potencia")
plt.grid()

plt.tight_layout()
plt.show()
