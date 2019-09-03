import numpy as np
import matplotlib.pyplot as plt


def export_rd_matrix_img(fname, rd_matrix, max_Doppler, dyn_range, dpi, interpolation='sinc', cmap='jet'):
    rd_matrix = 10 * np.log10(np.abs(rd_matrix) ** 2)
    rd_matrix -= np.max(rd_matrix)

    for i in range(np.shape(rd_matrix)[0]):  # Remove extreme low values
        for j in range(np.shape(rd_matrix)[1]):
            if rd_matrix[i, j] < -dyn_range:
                rd_matrix[i, j] = -dyn_range

    plt.ioff()

    rd_fig, rd_axes = plt.subplots()
    rd_plot = rd_axes.imshow(rd_matrix, interpolation=interpolation, cmap=cmap, origin='lower', aspect='auto')
    plt.ion()

    rd_axes.set_xlabel("Bistatic range cell")
    rd_axes.set_ylabel("Doppler Frequency [Hz]")
    rd_fig.colorbar(rd_plot)

    rd_axes.set_yticks(np.arange(0, np.size(rd_matrix, 0) + np.size(rd_matrix, 0) / 10, np.size(rd_matrix, 0) / 10))
    rd_axes.set_yticklabels(np.arange(-max_Doppler, max_Doppler + 2 * max_Doppler / 10, 2 * max_Doppler / 10))

    plt.tight_layout()
    rd_fig.savefig(fname, dpi=dpi)
    plt.close()