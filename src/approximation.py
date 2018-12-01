import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import util

def polynomial_basis(order, x):
	return_matrix = np.zeros((order+1,x.shape[0])) 
	for i in range(order+1):
		return_matrix[i,:] = x ** i

	return return_matrix.T


def trigonmetric_basis(order, x, period):
    return_matrix = np.zeros((2 * order, x.shape[0]))
    for i in range(1,order+1):
        return_matrix[2*(i-1),:] = np.cos(2 * np.pi/period * i * x) 
        return_matrix[2*(i-1)+1,:] =  np.sin(2 * np.pi/period * i * x)

    return return_matrix.T

def poly_with_trig_basis(order, x, period):
    poly_basis = polynomial_basis(order, x)
    trig_basis = trigonmetric_basis(order, x, period)
    return_matrix = np.hstack((poly_basis, trig_basis)) 
    return return_matrix


def least_sqaure_approximation(x, y, repeat, basis, order, period = 2):
    if basis == 'tri':
        A = trigonmetric_basis(order, x, period)
    elif basis == 'poly':
        A = polynomial_basis(order, x)
    elif basis == 'poly+trig':
        A = poly_with_trig_basis(order, x, period)

    stack_A = np.tile(A, (repeat, 1))

    coeff = np.linalg.lstsq(stack_A, y)[0]

    error = stack_A.dot(coeff) - y
    print("L_inf:",np.max(error))
    print("L_lsq:", np.sqrt(np.sum(error ** 2)))
    return error, coeff, A

if __name__ == '__main__':

    all_data = []
    for i in [1,2,3,5,6,7,8,9,10]:
        for j in range(1,4):
            data = util.load_one('../data/', 2, i, j)
            all_data.append(data)
    all_data = np.squeeze(np.array(all_data))
    #27 number of action x frame x 20 join x coordiante
    print(all_data.shape)
    general_curve = np.zeros(all_data.shape[1:])
    for i in range(20):
        for j in range(3):
            y = all_data[:,:,i,j] #action x frame
            y_flat = y.reshape(-1)
            x = np.arange(all_data.shape[1])
            error, coeff, basis = least_sqaure_approximation(x, y_flat, 27, 'poly+trig', 10)

            general_curve[:,i,j] = basis.dot(coeff)
    util.animate_skeleton(general_curve)
    # # import matplotlib.pyplot as plt
    # # plot_x = np.arange(data.shape[1], 0.01)
    # # # plt.plot(basis.dot(coeff), c='r')
    # # for i in range(27):
    # #     plt.plot(y[i,:],c='b')
    # # plt.show()

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')  
    # for i in range(y.shape[0]):
    #     ax.plot(y[i,:,0], y[i,:,1], y[i,:,2], '-b')
    # plt.show()