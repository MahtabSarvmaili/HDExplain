

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
        