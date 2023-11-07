from sklearn.manifold import TSNE


def explain_instance(train_dataset, test_tensor, test_pred, top_influencers):
    instances = []
    for i, test_feature in enumerate(test_tensor):
        feature_group = []
        label_group = []
        feature_group.append(test_feature)
        label_group.append(test_pred[i])
        train_indexes = top_influencers[i]
        for j in train_indexes:
            train_point = train_dataset[j]
            feature_group.append(train_point[0])
            label_group.append(train_point[1])
        instances.append([feature_group, label_group])

    return instances


def tsne(features, n_components = 2, verbose = 1, perplexity = 30, n_iter = 1000, metric = 'cosine'):
    tsne_results = TSNE(n_components=n_components,
                        verbose=verbose,
                        perplexity=perplexity,
                        n_iter=n_iter,
                        metric=metric).fit_transform(features)
    return tsne_results