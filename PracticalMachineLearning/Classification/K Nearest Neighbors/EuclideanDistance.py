from math import sqrt
import numpy as np
import matplotlib .pyplot as plt
from matplotlib import style
from collections import Counter
import warnings

style.use('fivethirtyeight')


# euclidean_distance = sqrt(
#   (plot1[0] - plot2[0]) ** 2 + (plot1[1] - plot2[1]) ** 2)

dataset = {'K': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
new_featues = [5, 7]


def K_nearest_neighbors(data, predict, K=3):
    if len(data) >= K:
        warnings.warn('K is set to a value less that total voting groups!')
    distances = []
    for group in data:
        for featues in data[group]:
            euclidean_distance = np.linalg.norm(
                np.array(featues) - np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:K]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result


result = K_nearest_neighbors(dataset, new_featues, K=3)
print(result)

for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0], ii[1], s=100, color=i)
plt.scatter(new_featues[0], new_featues[1], s=100, color = result)
plt.show()
