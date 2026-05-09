
import numpy as np
import matplotlib.pyplot as plt
import os
from pydub import AudioSegment
from scipy.signal import find_peaks

class analiseEspectral():
    @staticmethod
    def analiseporFourier(caminhoArquivo, nomeArquivo, amplitudeNormalizada = False):
        
        # Importando os arquivos
        arquivoAudio = os.path.join(caminhoArquivo, nomeArquivo)
        audio = AudioSegment.from_file(arquivoAudio).set_channels(1)

        # Extraindo dados e parametros
        taxadeAmostragem = audio.frame_rate
        periodo = 1 / taxadeAmostragem
        amostrasdeAudio = np.array(audio.get_array_of_samples())
        numerodeAlmostras = len(amostrasdeAudio)
        
        metadeAmostras = numerodeAlmostras // 2
        
        # Algoritmo pra FFT
        fftResultado = np.fft.fft(amostrasdeAudio)
        frequencias = np.fft.fftfreq(numerodeAlmostras, periodo)

        # Saidas
        frequenciasNyquist = frequencias[:metadeAmostras]
        amplitudes         = np.abs(fftResultado[:metadeAmostras]) * 2 / numerodeAlmostras

        if amplitudeNormalizada:
            soma_amp = np.sum(amplitudes)
            if soma_amp > 0:
                amplitudes = amplitudes / soma_amp
        return frequenciasNyquist, amplitudes
    
    @staticmethod
    def espectroAmplitude(frequenciasComplexas, amplitudes, frequenciadeCorte):

        plt.figure(figsize=(10, 4))
        plt.plot(frequenciasComplexas, amplitudes)
        plt.xlim(0, frequenciadeCorte)
        plt.title("Espectro de Amplitude do sinal")
        plt.xlabel("Frequências (Hz)")
        plt.ylabel("Amplitudes (u.a)")
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def selecionarPeaks(frequenciasComplexas, amplitudes, frequenciadeCorte, numeroAmostras):
        faixadeFrequencia = frequenciasComplexas <= frequenciadeCorte
        arrayFrequencias = frequenciasComplexas[faixadeFrequencia]
        arrayAmplitudes = amplitudes[faixadeFrequencia]

        indicesPicos, _ = find_peaks(arrayAmplitudes, distance=5)
        picosFrequencias = arrayFrequencias[indicesPicos]
        picosAmplitudes = arrayAmplitudes[indicesPicos]

        return picosFrequencias, picosAmplitudes