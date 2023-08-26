"""
I couldn't fit this one into a Jupyter notebook, because it uses
live microphone input. You'll need `libportaudio2` for this thing.

```bash
$ sudo apt install libportaudio2
```
"""

import matplotlib.animation as animation
import numpy as np
import scipy
import sounddevice as sd
from matplotlib import pyplot as plt


MAX_INT64_VALUE = 32767

SAMPLE_RATE = 44100  # Hz
SAMPLE_SPACING = 1.0 / SAMPLE_RATE  # seconds per sample

BLOCK_DURATION = 0.2  # seconds per block
BLOCK_LENGTH = int(SAMPLE_RATE * BLOCK_DURATION)  # samples per block, rounded down

MINIMUM_FREQUENCY = 82  # E2
MAXIMUM_FREQUENCY = 1318  # E6
FREQUENCIES = scipy.fft.rfftfreq(BLOCK_LENGTH, SAMPLE_SPACING)

FREQUENCY_FILTER = (FREQUENCIES <= MINIMUM_FREQUENCY) | (
    FREQUENCIES >= MAXIMUM_FREQUENCY
)

ABSOLUTE_THRESHOLD = 3
RELATIVE_THRESHOLD = 0.4  # % from max peak
HPS_HARMONICS_NUMBER = 4  # number of harmonics

block = np.zeros(1)

fig = plt.figure(figsize=(14, 8))


def get_hps(spectrum: np.ndarray) -> np.ndarray:
    """Perform harmonic product spectrum on a spectrum"""
    hps_spectrum = spectrum.copy()

    for harmonic in range(2, HPS_HARMONICS_NUMBER + 2):
        spectrum = scipy.signal.resample(spectrum, len(spectrum) // harmonic)
        hps_spectrum[: len(spectrum)] *= spectrum

    return hps_spectrum


def update_plot(*_) -> None:
    global block

    # normalize
    block = block.flatten() / MAX_INT64_VALUE
    block = block - np.mean(block)

    # pad to fit required shape
    block = np.pad(block, (0, BLOCK_LENGTH - len(block)))

    magnitudes = np.abs(scipy.fft.rfft(block))

    mask = (
        FREQUENCY_FILTER
        | (magnitudes < ABSOLUTE_THRESHOLD)
        | (magnitudes < (RELATIVE_THRESHOLD * np.max(magnitudes)))
    )
    magnitudes[mask] = 0.0

    hps_spectrum = get_hps(magnitudes)
    fundamental = FREQUENCIES[np.argmax(magnitudes)]

    fig.clf()
    axs = fig.subplots(3, 1)

    signal_plot = axs[0]

    signal_plot.plot(
        np.arange(BLOCK_LENGTH),
        block,
        color="b",
        label="Input Signal",
    )

    signal_plot.set_xlabel("Sample")
    signal_plot.set_ylabel("Amplitude")

    signal_plot.set_xlim(0, BLOCK_LENGTH * BLOCK_DURATION)
    signal_plot.set_ylim(-1, 1)
    signal_plot.legend()
    signal_plot.grid()

    magnitude_plot = axs[1]

    magnitude_plot.plot(
        FREQUENCIES,
        magnitudes,
        color="b",
        label="Fourier Transform",
    )
    magnitude_plot.axvline(
        fundamental,
        color="r",
        label="Fundamental Frequency",
    )

    magnitude_plot.set_xlabel("Frequency")
    magnitude_plot.set_ylabel("Magnitude")

    magnitude_plot.set_xticks(
        np.arange(MINIMUM_FREQUENCY, MAXIMUM_FREQUENCY + 1, 50),
    )
    magnitude_plot.set_xlim(MINIMUM_FREQUENCY, MAXIMUM_FREQUENCY)

    magnitude_plot.set_ylim(0, 100)
    magnitude_plot.legend()
    magnitude_plot.grid()

    hps_plot = axs[2]

    hps_plot.plot(
        FREQUENCIES,
        hps_spectrum,
        color="b",
        label="Harmonic Product Spectrum",
    )
    hps_plot.axvline(
        fundamental,
        color="r",
        label="Fundamental Frequency",
    )

    hps_plot.set_xlabel("Frequency")
    hps_plot.set_ylabel("Magnitude")

    hps_plot.set_xticks(
        np.arange(MINIMUM_FREQUENCY, MAXIMUM_FREQUENCY + 1, 50),
    )
    hps_plot.set_xlim(MINIMUM_FREQUENCY, MAXIMUM_FREQUENCY)

    hps_plot.set_ylim(-50, 50)
    hps_plot.legend()
    hps_plot.grid()


ani = animation.FuncAnimation(
    fig=fig,
    func=update_plot,
    frames=None,
    interval=BLOCK_DURATION * 1000,
    cache_frame_data=False,
)


def callback(indata: np.ndarray, *_) -> None:
    global block
    block = indata


with sd.InputStream(
    channels=1,
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_LENGTH,
    dtype=np.int16,
    latency="low",
    callback=callback,
) as stream:
    plt.show()
