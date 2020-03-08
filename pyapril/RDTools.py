import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def export_rd_matrix_img(fname, rd_matrix, max_Doppler, 
                         ref_point_range_index=0, 
                         ref_point_Doppler_index=0, 
                         box_color=None,
                         box_size=0,
                         dyn_range=None,
                         dpi=200, 
                         interpolation='sinc', 
                         cmap='jet'):
    """
        Description:
        ------------
        This function exports the given range-Doppler matrix into an image file.
        On the exported figure the X axis will represent the bistatic range while Y axis
        will show the Doppler frequencies. The Doppler ticks are automatically generated,
        but the caller should specifiy the maximum Doppler frequency in [Hz].
        
        If requested the function can draw an adition highligh box onto the exported
        range-Doppler map image. This feature is usefull when the user want to highlight
        a specific reflection. The color and the size of the highlight box is configurable
        via the 'box_color' and the 'box_size' paramters. The position of the box could be
        set by the By default, the highlight box is 
        not drawn. 
    
        In case the dynamic range paramter is not specified the function will automatically
        estimate the usefull dynamic range of the range-Doppler matrix.
        
        Parameters:
        -----------
        :param: fname                  : Filname into which the range-Doppler matrix will be exported
        :param: rd_matrix              : range-Doppler matrix to be exported
        :param: max_Doppler            : Maximum Doppler frequency in the range-Doppler matrix
        :param: ref_point_range_index  : (default: 0)
        :param: ref_point_Doppler_index: (default: 0)
        :param: box_color              : (default: None - Highlight box will not be drawn)
        :param: box_size               : (default: 0  - Highlight box will not be drawn)
        :param: dyn_range              : (default: None - Dynamic range will be automatically calculated)
        :param: dpi                    : (default: 200)
        :param: interpolation          : (default: 'sinc')
        :param: cmap                   : (default: 'jet')

            
        :type: fname                  : string
        :type: rd_matrix              : R x D complex numpy array
        :type: max_Doppler            : int
        :type: ref_point_range_index  : int
        :type: ref_point_Doppler_index: int
        :type: box_color              : string
        :type: box_size               : int
        :type: dyn_range              : float
        :type: dpi                    : int
        :type: interpolation          : string - matplotlib compatible e.q.:'sinc' / 'none'
        :type: cmap                   : string - matplotlib colormap e.g.:'jet' 

            
        Return values:
        --------------
        No return values
    """
    
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
    labels = []
    labels_float = np.arange(-max_Doppler, max_Doppler + 2 * max_Doppler / 10, 2 * max_Doppler / 10).tolist()
    for label_float in labels_float:
        labels.append("{:.1f}".format(label_float))
    rd_axes.set_yticklabels(labels)
    
    
    # Draw reference square if requested
    if box_color is not None and box_size != 0:
        rd_axes.add_patch(patches.Rectangle(
            (ref_point_range_index - box_size, ref_point_Doppler_index - box_size),   # (x,y)
            (box_size*2+1),          # width
            (box_size*2+1),          # height
            fill=False,
            edgecolor = box_color,
            linewidth = 1))

    plt.tight_layout()
    rd_fig.savefig(fname, dpi=dpi)
    plt.close()
    
