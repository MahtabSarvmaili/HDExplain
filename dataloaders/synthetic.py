import numpy as np
from sklearn.datasets import make_moons


def rectangular(n_samples, grid, n_classes, random_state=42):
    np.random.seed(random_state)
    x_split = np.linspace(-2, 2, num=grid[0]+1)
    y_split = np.linspace(-2,2, num=grid[1]+1)
    x_ranges = [(x_split[i],x_split[i+1]) for i in range(len(x_split)-1)]
    y_ranges = [(y_split[i],y_split[i+1]) for i in range(len(y_split)-1)]

    data_range = []
    for x_range in x_ranges:
        for y_range in y_ranges:
            data_range.append([x_range, y_range])

    n_regions = grid[0] * grid[1]

    region_labels = np.random.choice(n_classes, n_regions)
    x = np.random.uniform(-2, 2, n_samples)
    y = np.random.uniform(-2, 2, n_samples)
    features = np.stack([x,y]).T

    labels = []
    for d in features:
        for i,r in enumerate(data_range):
            if d[0]>=r[0][0] and d[0]<r[0][1] and d[1]>=r[1][0] and d[1]<r[1][1]:
                labels.append(region_labels[i])

    np.random.seed()
    return features, np.array(labels)

def moon(n_samples, n_classes=2, random_state=42):
    features, labels = make_moons(n_samples=n_samples, noise=0.1, 
                                  random_state=random_state)
    return features, labels