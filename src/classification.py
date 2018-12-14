import numpy as np 
from util import load_all, static_skeleton, joints
from approximation import do_approximation
from sklearn import svm
from confusion import polt_confusion_matrix_intergration

if __name__ == '__main__': 
    data, classes, subjects = load_all()
    print(data.shape)
    # ids = np.arange(data.shape[0])
    # np.random.shuffle(ids)

    train_x = data[1::2]
    train_y = classes[1::2]
    test_x = data[::2]
    test_y = classes[::2]
    features = []
    for class_id in range(1,21):
        data, coeff = do_approximation(train_x[np.where(train_y == class_id)])
        features.append(coeff.flatten())

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        coeff = np.square(coeff).sum(axis=-1).sum(axis=-1)
        data = data[50]

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for i in range(20):
        #     d = (coeff <= coeff[i]).sum()/20
        #     d = d*d
        #     print(d)
        #     ax.plot([data[i,0]], [data[i,1]], [data[i,2]], c='r',  marker="o", alpha=d)
        # for each in joints:
        #     joint_1 = each[0]
        #     joint_2 = each[1]
        #     graph, = ax.plot(data[[joint_1, joint_2],0], \
        #                     data[[joint_1, joint_2],1], \
        #                     data[[joint_1, joint_2],2], c= 'b')
        # ax.set_xlim3d([-1.0, 1.0])
        # ax.set_xlabel('X')
        # ax.set_ylim3d([-1.0, 1.0])
        # ax.set_ylabel('Y')
        # ax.set_zlim3d([-1.5, .5])
        # ax.set_zlabel('Z')
        # ax.set_title('3D Test')
        # plt.show()

    features = np.stack(features)
    total = 0
    confusion_matrix = np.zeros((20, 20))
    y_pred = []
    for i in range(test_x.shape[0]):
        data, coeff = do_approximation(np.reshape(test_x[i],(1,100,20,3)))
        coeff = coeff.flatten()
        dis = np.square(features - coeff).sum(axis=1)
        np.argsort(dis)[:3]
        confusion_matrix[np.argmin(dis), test_y[i]-1] += 1
        if (np.argmin(dis)+1 == test_y[i]):
            total += 1
        y_pred.append(np.argmin(dis)+1)
    polt_confusion_matrix_intergration(test_y, y_pred)
    print(total/test_x.shape[0])




