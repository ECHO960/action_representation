import numpy as np 
from util import load_all
from approximation import do_approximation
from sklearn import svm

if __name__ == '__main__': 
    data, classes, subjects = load_all()
    print(data.shape)
    ids = np.arange(data.shape[0])
    np.random.shuffle(ids)
    train_x = data[ids[:285]]
    train_y = classes[ids[:285]]
    test_x = data[ids[285:]]
    test_y = classes[ids[285:]]
    features = []
    for class_id in range(1,21):
        data, coeff = do_approximation(train_x[np.where(train_y == class_id)])
        features.append(coeff.flatten())
        # print(class_id)
    features = np.stack(features)
    total = 0
    for i in range(test_x.shape[0]):
        data, coeff = do_approximation(np.reshape(test_x[i],(1,100,20,3)))
        coeff = coeff.flatten()
        dis = np.square(features - coeff).sum(axis=1)
        if (np.argmin(dis)+1 == test_y[i]):
            total += 1
    print(total/test_x.shape[0])




