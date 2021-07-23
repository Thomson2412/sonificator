import math
import os
import soundfile as sf
import numpy as np


def extend(input_dir):
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            if ".wav" in filename or ".mp3" in filename:
                data, sample_rate = sf.read(file_path)
                channels = len(data.shape)
                length_s = len(data) / float(sample_rate)
                if length_s <= 5.0:
                    print(f"Start: {file_path}")
                    n = math.ceil(15 * sample_rate / len(data))
                    if channels == 2:
                        data = np.tile(data, (n, 1))
                    else:
                        data = np.tile(data, n)
                    sf.write(file_path, data, sample_rate)
                    print(f"End: {file_path}")


if __name__ == '__main__':
    extend("/mnt/datadrive/projects/thesis/Datasets/Audio/ESC-50-master")
