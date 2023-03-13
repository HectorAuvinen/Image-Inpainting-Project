
import os
import numpy as np

from matplotlib import pyplot as plt

from datasets import ex4


def plot(inputs, targets, predictions, path, update):
    """Plotting the inputs, targets and predictions to file `path`"""
    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    
    for i in range(len(inputs)):
        for ax, data, title in zip(axes, [inputs, targets, predictions], ["Input", "Target", "Prediction"]):
            ax.clear()
            ax.set_title(title)
            ax.imshow(data[i, 0], interpolation="none")
            #ax.imshow(np.transpose(data[i], (1, 2, 0))[:, :, 0:3], interpolation="none")
            #ax.imshow(np.transpose(data[i].astype("uint8"), (1, 2, 0))[:,:,0:3], interpolation="none")
            ax.set_axis_off()
        fig.savefig(os.path.join(path, f"{update:07d}_{i:02d}.png"), dpi=100)
    
    plt.close(fig)
 
'''
def plot(inputs, targets, predictions, path, update):
    """Plotting the inputs, targets and predictions to file `path`"""
    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

    for i in range(len(inputs)):
        for ax, data, title in zip(axes, [inputs, targets, predictions], ["Input", "Target", "Prediction"]):
            ax.clear()
            ax.set_title(title)
            ax.imshow(np.transpose(data[i].astype('uint8'), (1, 2, 0))[:, :, 0:3], interpolation="none", vmin=0,
                      vmax=255)
            ax.set_axis_off()
        fig.savefig(os.path.join(path, f"{update + 1:07d}_{i + 1:02d}.png"), dpi=100)

    plt.close(fig)
'''
#Get the target arrays from full predictions. 
#Input: model predictions, offsets & spacings from pickle file
#Output: predicted target arrays
def get_target_array(model_predictions,pkl_offsets,pkl_spacings):
    #create an empty list for the targets
    output_targets = []
    #loop over model predictions & apply ex4 function on each prediction with given offset and spacing, append to output targets list, and finally output list
    for i in range(len(model_predictions)):
        input_array,known_array,target_array = ex4(model_predictions[i],pkl_offsets[i],pkl_spacings[i])
        output_targets.append(np.array(target_array,dtype=np.uint8))
    return output_targets
