# import necessary modules
import numpy as np
from skimage import measure


def region_growing(mask_seed, mask_flood):
    """
    mask_seed: boolean mask from which region growing will be performed
    mask_flood: boolean mask which represents the maximum region growth
    """
    size = mask_flood.shape[0] // 2
    mask_seed = np.pad(mask_seed, ((size, size), (0, 0)), "wrap")
    mask_flood = np.pad(mask_flood, ((size, size), (0, 0)), "wrap")

    # binarize
    flood_labels = measure.label(mask_flood)
    seed_labels, num_seeds = measure.label(mask_seed, return_num=True)
    mask_out = np.zeros_like(mask_seed, dtype=np.bool)

    # iterate over all seeds in the seed mask
    for label in range(1, num_seeds + 1):
        # get all pixel locations of a single seed in the seed mask
        location = np.where(seed_labels == label)

        # for all pixel locations in the seed mask, check if their value in the flood image is nonzero
        flood_label = flood_labels[location]

        # if we have more than one overlapping pixel, we found a lesion
        # check which object value is the most common one
        if np.max(flood_label) > 0:
            flood_label = flood_label[np.nonzero(flood_label)]
            flood_label = np.argmax(np.bincount(flood_label))
        else:
            continue

        # get all pixel locations in the flood mask
        location_flood = np.where(flood_labels == flood_label)
        mask_out[location_flood] = True

    mask_out = mask_out[size:-size]
    return mask_out
