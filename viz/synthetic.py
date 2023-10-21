from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable


def data_viz_2d(X, y, name="synthetic_data"):

  fig, ax = plt.subplots(figsize=(4,4))
  # When the label y is 0, the class is represented with a blue square.
  # When the label y is 1, the class is represented with a green triangle.
  ax.plot(X[:, 0][y==2], X[:, 1][y==2], "*")
  ax.plot(X[:, 0][y==1], X[:, 1][y==1], "+")
  ax.plot(X[:, 0][y==0], X[:, 1][y==0], "gx")

  # X contains two features, x1 and x2
  plt.xlabel("")
  plt.ylabel("")

  # sns.kdeplot(x=X[:,0],y=X[:,1], bw_adjust=.4, fill=True, thresh=0, levels=100, cmap="YlOrBr", ax=ax)

  # Simplifying the plot by removing the axis scales.
  plt.xticks([])
  plt.yticks([])

  plt.tight_layout()

  # Displaying the plot.
  plt.savefig('plots/{0}'.format(name), format="pdf")


def ksd_distribution(ksd_scores, name="ksd_dist.pdf"):
  fig, ax = plt.subplots(figsize=(8,2))
  # When the label y is 0, the class is represented with a blue square.
  # When the label y is 1, the class is represented with a green triangle.
  sns.histplot(ksd_scores, ax=ax)

  # X contains two features, x1 and x2
  plt.xlabel(r"$\kappa_\theta(\mathbf{x}, \cdot)$", fontsize=16)
  plt.ylabel(r"Histogram", fontsize=16)
  plt.tight_layout()
  plt.savefig("plots/{0}".format(name), format='pdf')


def ksd_influence(X, y, x_test, y_test, score, name="ksd_influence.pdf"):

  fig, ax = plt.subplots(figsize=(4,4))
  # When the label y is 0, the class is represented with a blue square.
  # When the label y is 1, the class is represented with a green triangle.
  norm = plt.Normalize(-10, 10, clip=True)
  ax.scatter(X[:, 0][y==2], X[:, 1][y==2], c=score[y==2], norm=norm, marker="*", cmap='RdYlGn')
  ax.scatter(X[:, 0][y==1], X[:, 1][y==1], c=score[y==1], norm=norm, marker="+", cmap='RdYlGn')
  im=ax.scatter(X[:, 0][y==0], X[:, 1][y==0], c=score[y==0], norm=norm, marker="x", cmap='RdYlGn')


  test_marker = "*"
  if y_test==0:
    test_marker = "x"
  elif y_test == 1:
    test_marker = "+"
  else:
    test_marker = "*"

  ax.scatter(x_test[0], x_test[1], c='black', marker=test_marker,s=100)
  # X contains two features, x1 and x2
  plt.xlabel("")
  plt.ylabel("")

  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='5%', pad=0.05)

  # Simplifying the plot by removing the axis scales.
  ax.set_xticks([])
  ax.set_yticks([])

  fig.colorbar(im, cax=cax, orientation='vertical')

  # sns.kdeplot(x=X[:,0],y=X[:,1], bw_adjust=.4, fill=True, thresh=0, levels=100, cmap="YlOrBr", ax=ax)

  # Simplifying the plot by removing the axis scales.
  plt.xticks([])
  plt.yticks([])

  plt.tight_layout()

  plt.savefig("plots/{0}".format(name), format='pdf')