import numpy as np
import matplotlib.pyplot as plt


def im_convert(tensor):
  image = tensor.cpu().clone().detach().numpy()
  image = image.transpose(1, 2, 0)
  image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
  image = image.clip(0, 1)
  return image

def plot_explanation_images(instance, classes):
    n_images = len(instance[0])
    fig = plt.figure(figsize=(4 * n_images, 4))

    for idx in np.arange(n_images):
        ax = fig.add_subplot(1, n_images, idx+1, xticks=[], yticks=[])
        plt.imshow(im_convert(instance[0][idx]))
        ax.set_title(classes[instance[1][idx]])
    
    plt.show()