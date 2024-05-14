import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from PIL import Image


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
        ax.set_xlabel("Predict: {0}".format(classes[instance[1][idx]]), fontsize=32)
      else:
        ax.set_xlabel("Label: {0}".format(classes[instance[1][idx]]), fontsize=32)

  plt.tight_layout()
  plt.savefig("plots/{0}".format(name), format='pdf', bbox_inches='tight')
  plt.close()


def plot_images_in_2d(x, y, tensors, axis=None, zoom=1):
    if axis is None:
        axis = plt.gca()
    x, y = np.atleast_1d(x, y)
    for x0, y0, tensor in zip(x, y, tensors):
        image = Image.fromarray(im_convert(tensor))
        image.thumbnail((100, 100), Image.ANTIALIAS)
        img = OffsetImage(image, zoom=zoom)
        anno_box = AnnotationBbox(img, (x0, y0),
                                  xycoords='data',
                                  frameon=False)
        axis.add_artist(anno_box)
    axis.update_datalim(np.column_stack([x, y]))
    axis.autoscale()


def show_tsne(x, y, tensors):
    fig, axis = plt.subplots()
    fig.set_size_inches(22, 22, forward=True)
    plot_images_in_2d(x, y, tensors, zoom=0.3, axis=axis)
    plt.show()