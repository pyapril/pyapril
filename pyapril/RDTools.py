import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def export_rd_matrix_img(fname, rd_matrix, max_Doppler, dyn_range=None, dpi=200, interpolation='sinc', cmap='jet'):
    
    """
    --------------------------------
               Parameters
    --------------------------------
    """

    # Constant conversion parameters
    # -> Figure size
    w = 1920/250
    h = 1080/250
    
    # -> For dynamic range estimation
    range_cell_index = 20
    doppler_cell_index= 30 
    window_length=5
    window_width=5

    
    """
    --------------------------------
        Dynamic range compression
    --------------------------------
    """
    if dyn_range is None:
        # Noise floor estimation
        noise_floor = 0  # Cumulative sum of the environment power
        cell_counter = 0
        
        rd_matrix /= np.max(np.abs(rd_matrix))
        
        for wi in np.arange(-window_length, window_length + 1):
            for wj in np.arange(-window_width, window_width + 1):
                cell_counter += 1
                noise_floor += np.abs(rd_matrix[doppler_cell_index + wj, range_cell_index + wi]) ** 2
        
        noise_floor /= cell_counter  # Normalize for average calc
        dyn_range = -10*np.log10(noise_floor)

    
    rd_matrix = 10 * np.log10(np.abs(rd_matrix) ** 2)
    rd_matrix -= np.max(rd_matrix)

    for i in range(np.shape(rd_matrix)[0]):  # Remove extreme low values
        for j in range(np.shape(rd_matrix)[1]):
            if rd_matrix[i, j] < -dyn_range:
                rd_matrix[i, j] = -dyn_range
                
    """
    --------------------------------
            Display and save
    --------------------------------
    """

    plt.ioff()

    rd_fig, rd_axes = plt.subplots()
    rd_fig.set_size_inches(w,h)

    rd_plot = rd_axes.imshow(rd_matrix, interpolation=interpolation, cmap=cmap, origin='lower', aspect='auto')
    plt.ion()

    rd_axes.set_xlabel("Bistatic range cell")
    rd_axes.set_ylabel("Doppler Frequency [Hz]")
    rd_fig.colorbar(rd_plot)
    
    
    rd_axes.set_yticks(np.arange(0, np.size(rd_matrix, 0) + np.size(rd_matrix, 0) / 10, np.size(rd_matrix, 0) / 10))
    rd_axes.set_yticklabels(np.arange(-max_Doppler, max_Doppler + 2 * max_Doppler / 10, 2 * max_Doppler / 10))
    rd_axes.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    plt.tight_layout()
    rd_fig.savefig(fname, dpi=dpi)
    plt.close()
    
    
