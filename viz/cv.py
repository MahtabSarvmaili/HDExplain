import numpy as np
import matplotlib.pyplot as plt


def im_convert(tensor):
  image = tensor.cpu().clone().detach().numpy()
  image = image.transpose(1, 2, 0)
  image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
  image = image.clip(0, 1)
  return image

def plot_explanation_images(instance, classes, name=""):
  n_images = len(instance[0])
  fig = plt.figure(figsize=(4 * n_images, 4))

  for idx in np.arange(n_images):
      ax = fig.add_subplot(1, n_images, idx+1, xticks=[], yticks=[])
      plt.imshow(im_convert(instance[0][idx]))
      if idx == 0:
        ax.set_xlabel("Pred Class: {0}".format(classes[instance[1][idx]]), fontsize=18)
      else:
        ax.set_xlabel("Data Label: {0}".format(classes[instance[1][idx]]), fontsize=18)
    
  plt.tight_layout()
  plt.savefig("plots/{0}".format(name), format='pdf')