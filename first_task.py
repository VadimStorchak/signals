from datetime import datetime

import matplotlib.pyplot as plt
from numba import njit
import numpy

from data_extractor import data_extractor_wav

times, amplitudes = data_extractor_wav('Signals/MyAudio.wav')

@njit
def fourier_transform(omegas):
    """
        Прямое Преобразование Фурье по Методу прямоульников
    """
    delta_t = times[1] - times[0]
    X = numpy.zeros((len(omegas),), dtype=numpy.complex128)
    for i in range(len(omegas)):
        # print(f'Шаг {i} из {len(omegas)}')
        X[i] = numpy.dot(amplitudes, numpy.exp(-1j * omegas[i] * times))
    return X * delta_t


@njit
def inversion_fourier_transform(spectrum, omegas):
    """
        Обратное преобразование Фурье по Методу прямоульников
    """
    delta_omega = omegas[1] - omegas[0]
    x = numpy.zeros((len(times),))
    for i in range(len(times)):
        x[i] = numpy.dot(spectrum, numpy.exp(1j * omegas * times[i])).real
    return x * (delta_omega / numpy.pi)


def first_task(up_bounce, sampling_frequency):
    """
        Первое задание
    :param up_bounce: верхняя граница частотности
    :param sampling_frequency: частота дискретизации
    """
    f = numpy.arange(0, up_bounce, sampling_frequency)
    omegas = f * 2 * numpy.pi

    # сумма (прямоугольники)
    start = datetime.now()
    spectrum = fourier_transform(omegas)
    finish = datetime.now()
    print('Время работы Прямого преобразования Фурье: ' + str(finish - start))

    start = datetime.now()
    restored_signal = inversion_fourier_transform(spectrum, omegas)
    finish = datetime.now()
    print('Время работы Обратного преобразования Фурье ' + str(finish - start))

    plt.title(f'Восстановленный и Спектр при {up_bounce}Гц, {sampling_frequency}Гц')
    plt.subplot(3, 1, 1).plot(times, amplitudes, color='black', label="Исходный")
    plt.subplot(3, 1, 2).plot(times, amplitudes, 'b', label="Исходный", linestyle='--')
    plt.subplot(3, 1, 2).plot(times, restored_signal, 'r', label="Восстановленный")
    plt.subplot(3, 1, 3).plot(omegas / (2 * numpy.pi), abs(spectrum), '', label="Спектр")

    plt.show()


if __name__ == "__main__":
    print('Первая лабораторная работа')

    # График исходного сигнала
    plt.plot(times, amplitudes, color='black')
    plt.xlabel('Время, с')
    plt.ylabel('Амплитуда')
    plt.title('График сигнала')
    plt.show()

    # Изменение шага дискретизации
    first_task(20_000, 0.1)
