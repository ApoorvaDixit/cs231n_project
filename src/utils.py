
import time
import pytz
from datetime import datetime
import matplotlib.pyplot as plt
import torch

from data.dataset_utils import get_label


def viz(tile):
    image = tile['tile']

    red_band = image[0, 0, :, :]
    green_band = image[0, 1, :, :]
    blue_band = image[0, 2, :, :]
    infrared_band = image[0, 3, :, :]

    # Combining the color bands into an RGB image
    rgb_image = torch.stack([red_band, green_band, blue_band], dim=2)

    # Plotting the RGB image and infrared band
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'True label: {get_label(tile["label"])}')
    axs[0].imshow(rgb_image)
    axs[0].set_title('RGB Image')
    axs[0].axis('off')
    axs[1].imshow(infrared_band, cmap='gray')
    axs[1].set_title('Infrared Band')
    axs[1].axis('off')

    # Display the plot
    plt.tight_layout()
    plt.show()

def get_timestr():
    t0 = time.time()
    pst = pytz.timezone('America/Los_Angeles')
    datetime_pst = datetime.fromtimestamp(t0, pst)
    return datetime_pst.strftime("%I:%M %p, %B %d")