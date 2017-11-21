from math import sqrt
import numpy as np
from collections import Counter
import warnings
import pandas as pd
import random


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


df = pd.read_csv('BreastCancerData.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}

train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]


for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = K_nearest_neighbors(train_set, data, K=5)
        if group == vote:
            correct += 1
        total += 1

print('accuracy:', correct / total)
