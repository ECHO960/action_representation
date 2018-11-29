import numpy as np
import matplotlib.pyplot as plt
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
    data, _, _ = util.load_all('../data/')

    y = data[3:6,:,0,0]
    y_flat = y.reshape(-1)
    x = np.arange(data.shape[1])
    error, coeff, basis = least_sqaure_approximation(x, y_flat, 3, 'poly+trig', 10)

    import matplotlib.pyplot as plt
    plot_x = np.arange(data.shape[1], 0.01)
    plt.plot(basis.dot(coeff), c='r')
    plt.plot(y[0,:],c='b')
    plt.plot(y[1,:],c='y')
    plt.plot(y[2,:],c='g')
    plt.show()