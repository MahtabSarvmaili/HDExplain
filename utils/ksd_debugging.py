import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.2023, 0.1994, 0.2010)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


def process_kernel_debugging(
        train_loader, dt_i, dt_ids, stein_debug, threshold, class_names, name):

    df = pd.DataFrame.from_dict(stein_debug)
    df.sort_values(by="ksd", ascending=False, inplace=True)
    df.set_index(dt_ids[df.index], inplace=True)
    dissim = df['scalars'] < threshold
    sim = df['scalars'] >= threshold
    df_idx = df.index.to_numpy()

    least_dissim = df_idx[dissim][0]
    most_dissim = df_idx[dissim][-1]
    least_sim = df_idx[sim][-2]
    most_sim = df_idx[sim][0]
    points_to_plot = [dt_i, most_sim, least_sim, least_dissim, most_dissim]

    titles = ['Original', 'Most similar', 'Least similar', 'Least dissimilar', 'Most dissimilar']

    instances = []
    feature_group = []
    label_group = []

    for j in points_to_plot:
        train_point = train_loader.dataset[j]
        feature_group.append(train_point[0])
        label_group.append(train_point[1])
    instances.append(feature_group)
    instances.append(label_group)

    n_images = len(instances[0])
    fig = plt.figure(figsize=(4 * n_images, 4))

    for id in np.arange(n_images):
        ax = fig.add_subplot(1, n_images, id + 1, xticks=[], yticks=[])
        plt.imshow(im_convert(instances[0][id]))
        ax.set_xlabel("{0} \n Label: {1}".format(titles[id], class_names[instances[1][id]]), fontsize=28)
    plt.tight_layout()
    plt.savefig("plots/{0}".format(name), format='pdf', bbox_inches='tight', pad_inches=0)
