from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pylab
from numba import njit

from data_extractor import data_extractor_wav

times, amplitudes = data_extractor_wav('Signals/MyAudio.wav')

@njit
def detailed_fourier_transform():
    """
        Прямое Дискретное Преобразование Фурье
    """
    N = len(amplitudes)
    X = np.zeros((N,), dtype=np.complex128)
    n = np.arange(N)
    for k in range(N):
        e = np.exp(-2j * np.pi * k * n / N)
        X[k] = np.dot(amplitudes, e)
    return X / np.sqrt(N)


@njit
def inversion_detailed_fourier_transform(spectrum):
    """
        Обратное Дискретное Преобразование Фурье
    """
    N = len(spectrum)
    restored_signal = np.zeros((N,), dtype=np.complex128)
    k = np.arange(N)
    for n in range(N):
        e = np.exp(2j * np.pi * k * n / N)
        restored_signal[n] = np.dot(spectrum, e)
    return restored_signal / np.sqrt(N)


def second_task_dft():
    """
        Дискретное Преобразование, Вычисление и вывод графика
    """
    n = len(amplitudes)
    delta_t = times[1] - times[0]
    delta_omega = 2 * np.pi / (n * delta_t)
    omegas = np.array([k * delta_omega for k in range(n)])

    start = datetime.now()
    spectrum = detailed_fourier_transform()
    stop = datetime.now()
    print(f'Дискретное преобразование Фурье завершено за {str(stop - start)}')

    start = datetime.now()
    restored_signal = inversion_detailed_fourier_transform(spectrum)
    stop = datetime.now()
    print(f'Обратное дискретное преобразование Фурье завершено за {str(stop - start)}')

    ax1 = fig.add_subplot(3, 1, 3)
    ax1.plot(omegas / (2 * np.pi), abs(spectrum), color='green', label="Спектр Дискретное преобразование")

    ax1.set_xlabel('Частота')
    ax1.set_ylabel('Амплитуда')

    ax.plot(times, restored_signal, 'r', label="Восстановленный сигнал", linestyle='--')

    ax.legend()
    ax1.legend()


def second_task_fft():
    """
        Быстрое Преобразование, вычисление и вывод графика
    """

    start = datetime.now()
    spectrum = np.fft.fft(amplitudes, norm='ortho')
    stop = datetime.now()
    print(f'Быстрое преобразование Фурье завершено за {str(stop - start)}')

    start = datetime.now()
    restored_signal = np.fft.ifft(spectrum, norm='ortho')
    stop = datetime.now()
    print(f'Обратное Быстрое преобразование Фурье завершено за {str(stop - start)}')


    n = len(amplitudes)
    delta_t = times[1] - times[0]
    delta_omega = 2 * np.pi / (n * delta_t)
    omegas = np.array([k * delta_omega for k in range(n)])

    ax1 = fig.add_subplot(3, 1, 3)
    ax1.plot(omegas / (2 * np.pi), abs(spectrum), color='green', label="Спектр Быстрое преобразование")
    ax1.set_xlabel('Частота')
    ax1.set_ylabel('Амплитуда')

    ax.plot(times, restored_signal, color='red', label="Восстановленный сигнал", linestyle='--')

    ax.legend()
    ax1.legend()


if __name__ == "__main__":
    print('Вторая лабораторная работа')

    fig = pylab.figure(1)
    ax = fig.add_subplot(3, 1, 1)
    ax.plot(times, amplitudes, color='black', label="Исходный сигнал")
    ax.set_xlabel("Время, с")
    ax.set_ylabel('Амплитуда')

    ax = fig.add_subplot(3, 1, 2)
    ax.plot(times, amplitudes, color='green', label="Исходный сигнал")
    ax.set_xlabel("Время, c")
    ax.set_ylabel('Амплитуда')

    # second_task_dft()
    second_task_fft()

    plt.show()
