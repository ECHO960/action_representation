import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import util
from relative_feature import relative_joint_transform, relative_joint_transform_rev,\
        relative_time_transform, relative_time_transform_rev

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

def do_approximation(data):
    all_data = relative_time_transform(data)
    general_curve = np.zeros(all_data.shape[1:])
    shape = (1, general_curve.shape[0], general_curve.shape[1], general_curve.shape[2])
    final_coeff = np.zeros((shape[2], shape[3], 11))
    for i in range(shape[2]):
        for j in range(shape[3]):
            y = all_data[:,1:,i,j] #action x frame
            y_flat = y.reshape(-1)
            x = np.arange(all_data.shape[1]-1)
            error, coeff, basis = least_sqaure_approximation(x, y_flat, all_data.shape[0], 'poly', 10)
            general_curve[1:,i,j] = basis.dot(coeff)
            final_coeff[i,j,:] = coeff
    general_curve[0,:,:] = all_data[:,0,:,:].mean(axis=0)
    general_curve = relative_time_transform_rev(np.reshape(general_curve, shape))[0]
    return general_curve, final_coeff

def do_approximation_normal(data):
    all_data = data
    general_curve = np.zeros(all_data.shape[1:])
    shape = (1, general_curve.shape[0], general_curve.shape[1], general_curve.shape[2])
    final_coeff = np.zeros((shape[2], shape[3], 2))
    for i in range(shape[2]):
        for j in range(shape[3]):
            y = all_data[:,:,i,j] #action x frame
            y_flat = y.reshape(-1)
            x = np.arange(all_data.shape[1])
            error, coeff, basis = least_sqaure_approximation(x, y_flat, all_data.shape[0], 'poly', 1)
            general_curve[:,i,j] = basis.dot(coeff)
            final_coeff[i,j,:] = coeff
    # general_curve[0,:,:] = all_data[:,0,:,:].mean(axis=0)
    # general_curve = relative_time_transform_rev(np.reshape(general_curve, shape))[0]
    return general_curve, final_coeff


def draw_approximation_curve_naive():
    """
    sequence - 1 x 
    """
    
    all_data = []
    for i in [1,2,3,5,6,7,8,9,10]:
        for j in range(1,4):
            data = util.load_one('../data/', 1, i, j)
            all_data.append(data)
    y = np.squeeze(np.array(all_data)) #sample x frame x 20 x coordinate
    joint = 8
    coordinate = 2      
    coordinate_str = 'xyz'
    import matplotlib.pyplot as plt
    x = np.arange(y.shape[1])
    # plt.plot(basis.dot(coeff), c='r')

    y_flat = y[:,:,joint,coordinate].reshape(-1) #action x frame
    error, coeff, basis = least_sqaure_approximation(x, y_flat, y.shape[0], 'poly', 10)
    general_curve = basis.dot(coeff)

    for i in range(27):
        if i == 0:
            plt.plot(x, y[i,:, joint, coordinate],c='b', label='Sample curves')
        else:
            plt.plot(x, y[i,:, joint, coordinate],c='b')
    plt.grid(True)
    plt.plot(x, general_curve, c='r',linewidth='5', label='Approximated curve')
    plt.legend()
    plt.title("Naive method: Trajectory of joint %d coordinate %s" % (joint, coordinate_str[coordinate]))
    plt.xlabel('Frame')
    plt.ylabel('Position')
    plt.show()

def draw_approximation_curve_temporal_difference(joint):
    """
    sequence - 1 x 
    """
    
    all_data = []
    for i in [1,2,3,5,6,7,8,9,10]:
        for j in range(1,4):
            data = util.load_one('../data/', 1, i, j)
            all_data.append(data)
    y = np.squeeze(np.array(all_data)) #sample x frame x 20 x coordinate
    y = relative_time_transform(y)[:,1:,:,:]
    coordinate = 2
    coordinate_str = 'xyz'
    import matplotlib.pyplot as plt
    x = np.arange(y.shape[1])
    # plt.plot(basis.dot(coeff), c='r')

    y_flat = y[:,:,joint,coordinate].reshape(-1) #action x frame
    error, coeff, basis = least_sqaure_approximation(x, y_flat, y.shape[0], 'poly', 10)
    general_curve = basis.dot(coeff)

    for i in range(27):
        if i == 0:
            plt.plot(x, y[i,:, joint, coordinate],c='b', label='Sample curves')
        else:
            plt.plot(x, y[i,:, joint, coordinate],c='b')
    plt.grid(True)
    plt.plot(x, general_curve, c='r',linewidth='5', label='Approximated curve')
    plt.legend()
    plt.title("Temporal difference method: Trajectory of joint %d coordinate %s" % (joint, coordinate_str[coordinate]))
    plt.xlabel('Frame')
    plt.ylabel('Position')
    plt.show()

def generate_animation():
    all_data = []
    for i in [1,2,3,5,6,7,8,9,10]:
        for j in range(1,4):
            data = util.load_one('../data/', 2, i, j)
            all_data.append(data)
    all_data = np.squeeze(np.array(all_data))
    general_curve, final_coeff = do_approximation(all_data)
    util.animate_skeleton(general_curve, 'temporal_difference.mp4', caption="temporal difference method")

    


if __name__ == '__main__':
    # all_data = []
    # data, classes, subjects = util.load_all()
    # for i in range(1,21):
    #     print("Class : ",i)
    #     all_data = data[np.where(classes == i)]
    #     print(np.isnan(all_data).any())
    #     # general_curve, final_coeff = do_approximation(all_data)
    #     print("============Class : ",i)
    # util.animate_skeleton(general_curve)
    # draw_approximation_curve_naive()
    generate_animation()
    # for i in range(20):
    #     draw_approximation_curve_temporal_difference(i)

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
