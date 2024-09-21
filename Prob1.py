
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import *

# preprocess and load data
filepath = 'train_data.pkl'
if not os.path.exists(filepath):
    general_process(output_filepath=filepath)
with open(filepath, 'rb') as fin:
    df = pickle.load(fin)
    
x = [i for i in range(1024)]

# Filter the data for '正弦波' waveforms
sine_wave_data = df[df['type_waveform'] == '正弦波'].head().iloc[0, -1]
triangle_wave_data = df[df['type_waveform'] == '三角波'].head().iloc[0, -1]
trapezoidal_wave_data = df[df['type_waveform'] == '梯形波'].head().iloc[2, -1]

# Plot the waveforms on the same figure
plt.plot(x, sine_wave_data, label='正弦波')
plt.plot(x, triangle_wave_data, label='三角波')
plt.plot(x, trapezoidal_wave_data, label='梯形波')
plt.scatter(x, sine_wave_data, label='正弦波')
plt.scatter(x, triangle_wave_data, label='三角波')
plt.scatter(x, trapezoidal_wave_data, label='梯形波')
plt.title('Waveform Plot for Different Waveforms')
plt.xlabel('Sample Points')
plt.ylabel('Magnetic Flux Density (B, T)')
plt.legend()
plt.show()
