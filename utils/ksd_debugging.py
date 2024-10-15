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


def process_kernel_debugging(train_loader, model, dt_i, dt_ids, stein_debug, ksd_threshold=0.65, threshold=0.6,
                             class_names=None, name="", device="cuda", show_pred=False):

    df = pd.DataFrame.from_dict(stein_debug)
    df.sort_values(by="ksd", ascending=False, inplace=True)
    df.set_index(dt_ids[df.index], inplace=True)
    df = df.drop(df.index[0])
    dissim = df['scalars'][df["ksd"] > ksd_threshold] < threshold
    df_ = df[df["ksd"] > ksd_threshold][dissim]
    df_.sort_values(by="ksd", ascending=False, inplace=True)
    df_idx = df_.index.to_numpy()
    points_to_plot = list(df_idx[:4])
    points_to_plot.insert(0, dt_i)
    instances = []
    feature_group = []
    label_group = []
    pred_group = []

    for j in points_to_plot:

        train_point = train_loader.dataset[j]
        feature_group.append(train_point[0])
        label_group.append(train_point[1])
        img = train_point[0].unsqueeze(0)
        pred_group.append(model.predict(img).argmax(dim=1).item())

    instances.append(feature_group)
    instances.append(label_group)
    instances.append(pred_group)


    n_images = len(instances[0])
    fig = plt.figure(figsize=(4 * n_images, 4))

    for id in np.arange(n_images):
        ax = fig.add_subplot(1, n_images, id + 1, xticks=[], yticks=[])
        plt.imshow(im_convert(instances[0][id]))
        if id == 0:
            ax.set_ylabel("Original", fontsize=30)
            if show_pred:
                ax.set_xlabel("Label: {0} \n Pred: {1}".format(class_names[instances[1][id]], class_names[instances[2][id]]), fontsize=28)
            else:
                ax.set_xlabel("Label: {0}".format(class_names[instances[1][id]]), fontsize=30)
    plt.tight_layout()
    plt.savefig("plots/{0}".format(name), format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

    df.sort_values(by="ksd", ascending=False, inplace=True)
    ksd_sort_idx = df.iloc[0].name

    df.sort_values(by="rbf_kernel", ascending=False, inplace=True)
    rbf_sort_idx = df.iloc[0].name

    df.sort_values(by="scalars", ascending=False, inplace=True)
    scalars_sort_idx = df.iloc[0].name

    titles = ["Original", "RBF", "$IF^{*}$", "KSD"] if show_pred else ["Original", "RBF", "IF", "KSD"]

    points_to_plot = [dt_i, rbf_sort_idx, scalars_sort_idx, ksd_sort_idx]
    instances = []
    feature_group = []
    label_group = []
    pred_group = []

    for j in points_to_plot:

        train_point = train_loader.dataset[j]
        feature_group.append(train_point[0])
        label_group.append(train_point[1])
        img = train_point[0].unsqueeze(0)
        pred_group.append(model.predict(img).argmax(dim=1).item())

    instances.append(feature_group)
    instances.append(label_group)
    instances.append(pred_group)

    a = name.split(".")
    a[0] = a[0] + "_comp"
    name = ".".join(a)

    n_images = len(instances[0])
    fig = plt.figure(figsize=(3.7 * n_images, 4))

    for id in np.arange(n_images):
        ax = fig.add_subplot(1, n_images, id + 1, xticks=[], yticks=[])
        plt.imshow(im_convert(instances[0][id]))
        ax.set_xlabel("Label: {0} \n Pred: {1}".format(class_names[instances[1][id]], class_names[instances[2][id]]), fontsize=28)
        ax.set_title(titles[id], fontsize=30)
    plt.tight_layout()
    pad_inch = 0.1 if show_pred else 0
    plt.savefig("plots/{0}".format(name), format='pdf', bbox_inches='tight', pad_inches=pad_inch)
    plt.close()


def index_to_debug(model, train_loader, indices, explainer, class_names, name_template, ksd_threshold, threshold, args):

    for i in indices:
        X_test, y_test = train_loader.dataset.__getitem__(i)
        X_test = X_test.unsqueeze(dim=0).cpu().numpy()
        y_test = np.array(y_test).reshape(1)
        influence_scores = explainer.ksd_debug(train_loader, X_test, y_test)
        dt_ids = influence_scores.argsort()[::-1].reshape(-1)
        ksd_eval_res = explainer.datapoint_model_discrepancy(train_loader, X_test, y_test, i, dt_ids)
        process_kernel_debugging(train_loader, model, i, dt_ids, ksd_eval_res, ksd_threshold, threshold, class_names,
                                 name_template.format(args.data, i))