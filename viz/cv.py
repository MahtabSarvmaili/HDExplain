import numpy as np
import matplotlib.pyplot as plt


def im_convert(tensor):
  image = tensor.cpu().clone().detach().numpy()
  image = image.transpose(1, 2, 0)
  image = image * np.array((0.2023, 0.1994, 0.2010)) + np.array((0.485, 0.456, 0.406))
  image = image.clip(0, 1)
  return image

def plot_explanation_images(instance, classes, name=""):
  n_images = len(instance[0])
  fig = plt.figure(figsize=(4 * n_images, 4))

  for idx in np.arange(n_images):
      ax = fig.add_subplot(1, n_images, idx+1, xticks=[], yticks=[])
      plt.imshow(im_convert(instance[0][idx]))
      if idx == 0:
        ax.set_xlabel("Predict: {0}".format(classes[instance[1][idx]]), fontsize=28)
      else:
        ax.set_xlabel("Label: {0}".format(classes[instance[1][idx]]), fontsize=28)
    
  plt.tight_layout()
  plt.savefig("plots/{0}".format(name), format='pdf')