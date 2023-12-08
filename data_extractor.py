import wave

import numpy


def data_extractor_wav(path):
    """
        Получение информации из аудио файла. !!!Стерео!!!
    :param path: путь до файла
    :return: times - фрагменты временных промежутков
    :return: amplitudes - амплитуда сигнала в момент времени
    """
    audio_file = wave.open(path, mode='r')

    (number_channels, sampling_width, frame_rate, frames, compile_type, compile_name) = audio_file.getparams()
    content = audio_file.readframes(frames)

    types = {
        1: numpy.int8,
        2: numpy.int16,
        4: numpy.int32
    }

    samples = numpy.fromstring(content, dtype=types[sampling_width])
    amplitudes = []
    for i in range(0, len(samples), 2):
        amplitudes.append((samples[i] + samples[i + 1]) / 2)
    times = []
    for i in range(1, frames + 1):
        times.append(i / frame_rate)
    times = numpy.array(times)
    amplitudes = numpy.array(amplitudes, dtype=numpy.complex128)
    return times, amplitudes
