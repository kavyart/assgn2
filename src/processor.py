import matplotlib.pyplot as plt
import numpy as np
import math
from skimage import io
# from skimage import color
# from scipy import interpolate

from cp_hw2 import writeHDR, read_colorchecker_gm

####### WEIGHTING SCHEMES #######
Z_min = 0.05
Z_max = 0.95

def w_uniform_func(z):
    if ((Z_min <= z) and (z <= Z_max)):
        return 1
    else:
        return 0
w_uniform = np.vectorize(w_uniform_func)

def w_tent_func(z):
    if ((Z_min <= z) and (z <= Z_max)):
        return min(z, 1-z)
    else:
        return 0
w_tent = np.vectorize(w_tent_func)

def w_gaussian_func(z):
    if ((Z_min <= z) and (z <= Z_max)):
        return math.exp(-16 * ((z-0.5)**2))
    else:
        return 0
w_gaussian = np.vectorize(w_gaussian_func)

def w_photon_func(z, t_k):
    if ((Z_min <= z) and (z <= Z_max)):
        return t_k
    else:
        return 0
w_photon = np.vectorize(w_photon_func)


####### LINEARIZE RENDERED IMAGES #######
def linearize_rendered_images(jpg, t):
    vector_g = np.load('g_jpg_tent.npy')
    return np.exp(vector_g[jpg])
    Z = jpg

    # Initialize known matrices
    n = 256
    A = np.zeros((Z.size + n, n + Z.shape[0]))
    b = np.zeros(A.shape[0])

    w = w_photon

    idx = 0
    for i in range(Z.shape[0]):
        for k in range (Z.shape[2]):
            for c in range(Z.shape[1]):
                wij = w(Z[i,c,k] / 255)
                A[idx,Z[i,c,k]+1] = wij
                A[idx,n+i] = -wij
                b[idx] = wij * np.log(t[k])
                idx += 1

    A[idx,129] = 1
    idx += 1

    l = 100
    for i in range(n-2):
        wi = w((i+1) / 255)
        A[idx,i] = l * wi
        A[idx,i+1] = -2 * l * wi
        A[idx,i+2] = l * wi
        idx += 1

    # Solve the least-squares optimization problem
    v = np.linalg.lstsq(A, b, rcond=None)[0]
    vector_g = v[0:n]
    print(vector_g)
    fig = plt.figure()
    plt.plot(vector_g)
    plt.xlabel("Pixel Value")
    plt.ylabel("Exposure")
    plt.savefig('gplot.png')
    plt.show()

    # g = np.vectorize(vector_g)

    I_lin = np.exp(vector_g[Z])
    print(I_lin)

    # np.save('g_jpg_photon.npy', vector_g)

    return I_lin

    # g = np.vectorize(add)
    # arr = g(arr)


####### MERGING EXPOSURE STACK INTO HDR IMAGE #######
def merge_exposure_stack(I_LDR, I_lin, t, width, height, merge='lin'):
    w = w_tent
    epsilon = 0.01

    numerator = np.zeros((height, width, 3))
    denominator = np.zeros((height, width, 3))

    if (merge == 'lin'):
        for k in range(I_LDR.shape[2]):
            i_ldr = I_LDR[:,:,k].reshape((height, width, 3))
            i_lin = I_lin[:,:,k].reshape((height, width, 3))
            numerator += w(i_ldr/255) * i_lin / t[k]
            denominator += w(i_ldr/255)
        I_HDR = np.nan_to_num(numerator / denominator)
    else: # merge == 'log'
        for k in range(I_LDR.shape[2]):
            i_ldr = I_LDR[:,:,k].reshape((height, width, 3))
            i_lin = I_lin[:,:,k].reshape((height, width, 3))
            numerator += w(i_ldr/255) * (np.log(i_lin + epsilon) - np.log(t[k]))
            denominator += w(i_ldr/255)
        I_HDR = np.nan_to_num(np.exp(numerator / denominator))

    # lin_numerator = np.sum(w(I_LDR) * I_lin / t, axis=-1)
    # log_numerator = np.sum(w(I_LDR) * (np.log(I_lin + epsilon) - np.log(t)), axis=-1)
    # denominator = np.sum(w(I_LDR), axis=-1)

    # I_HDR_lin = num / den
    # print(I_HDR_lin)
    # I_HDR_lin = np.nan_to_num(I_HDR_lin)

    writeHDR('tent' + merge + '.HDR', I_HDR)
    # print(I_HDR_lin)
    # I_HDR_log = np.exp(log_numerator / denominator)
    # print(I_HDR_log)
    # I_HDR_log = np.nan_to_num(I_HDR_log)
    # print(I_HDR_log)
    # fig = plt.figure()
    # # plt.plot(I_HDR_lin)
    # # plt.xlabel("Pixel Value")
    # # plt.ylabel("Exposure")
    # # plt.savefig('gplot.png')
    # # plt.show()
    # plt.imshow(I_HDR_lin)
    # print(I_HDR_lin.shape)
    return I_HDR



####### COLOR CORRECTION AND WHITE BALANCING #######
                     #  x_min y_min  x_max y_max
colorchecker_squares = np.array(  [ (3315, 1421, 3422, 1541),
                                    (3478, 1427, 3574, 1528),
                                    (3629, 1421, 3737, 1534),
                                    (3774, 1421, 3888, 1522),

                                    (3308, 1263, 3410, 1365),
                                    (3478, 1269, 3574, 1371),
                                    (3623, 1263, 3719, 1365),
                                    (3780, 1263, 3876, 1359),

                                    (3308, 1112, 3416, 1220),
                                    (3460, 1106, 3561, 1208),
                                    (3623, 1106, 3719, 1202),
                                    (3780, 1106, 3864, 1202),

                                    (3296, 949, 3398, 1045),
                                    (3454, 943, 3555, 1051),
                                    (3611, 943, 3701, 1045),
                                    (3768, 955, 3858, 1051),

                                    (3290, 785, 3392, 887),
                                    (3454, 785, 3549, 875),
                                    (3605, 779, 3701, 887),
                                    (3768, 798, 3858, 887),
                    
                                    (3290, 628, 3362, 712),
                                    (3454, 628, 3525, 706),
                                    (3611, 634, 3682, 706),
                                    (3762, 634, 3834, 718) ] )

def colorchecker_averages(I_HDR, colorchecker_coords, width, height):
    A = np.zeros((3*24,12))
    b = np.zeros(3*24)
    (red, green, blue) = read_colorchecker_gm()
    red = red.flatten()
    green = green.flatten()
    blue = blue.flatten()
    for i in range(colorchecker_coords.shape[0]):
        (x_min, y_min, x_max, y_max) = colorchecker_coords[i]
        # x_min += 3
        # y_min += 3
        # x_max -= 3
        # y_max -= 3
        square = I_HDR[y_min:y_max, x_min:x_max, :]
        # (r, g, b) = np.mean(I_HDR[y_min:y_max, x_min:x_max, :], axis=(0,1))
        r_mean = np.mean(square[:,:,0])
        g_mean = np.mean(square[:,:,1])
        b_mean = np.mean(square[:,:,2])
        for j in range(3):
            A[3*i+j][4*j] = r_mean
            A[3*i+j][4*j+1] = g_mean
            A[3*i+j][4*j+2] = b_mean
            A[3*i+j][4*j+3] = 1
        b[3*i] = red[i]
        b[3*i+1] = green[i]
        b[3*i+2] = blue[i]

    
    # b = np.stack((red.flatten(), green.flatten(), blue.flatten()), axis=-1).flatten()
    print('aaaaa')
    print(red.shape)
    print(colorchecker_coords.shape)
    # print(A)
    # print(red)
    # print(green)
    # print(blue)
    # print(b)
    print('bbbbb')

    # rgb_transform = np.linalg.lstsq(image_rgb, real_rgb, rcond=None)
    rgb_transform = np.linalg.lstsq(A, b, rcond=None)[0].reshape(3,4)

    # print(I_HDR)
    homogenous_hdr = np.insert(I_HDR, 3, 1, axis=2)
    # print(homogenous_hdr)

    print(I_HDR.shape)
    print(homogenous_hdr.shape)
    new_hdr = np.zeros((height, width, 3))
    new_hdr = np.dot(rgb_transform, homogenous_hdr.reshape(width*height,4).swapaxes(0,1)).swapaxes(1,0).reshape(height,width,3)
    # for i in range(width):
    #     for j in range(height):
    #         new_hdr[j][i] = np.dot(rgb_transform, homogenous_hdr[j][i])
    # new_hdr = new_hdr / 2
    # new_hdr = np.dstack((new_hdr[:,:,2],new_hdr[:,:,1],new_hdr[:,:,0]))
    new_hdr = np.clip(new_hdr, 0, 100)
    print(new_hdr)
    print(new_hdr.shape)

    writeHDR('colorcorrected.HDR', new_hdr)


    



    # return I_HDR


def white_balancing(I_HDR):
    N = I_HDR.shape[0] * I_HDR.shape[1]
    K = 0.15
    B = 0.95
    epsilon = 0.01

    I_mHDR = np.exp((1/N) * np.sum(np.log(I_HDR + epsilon)))
    I_tildeHDR = K * (I_HDR / I_mHDR)
    I_white = B * np.max(I_tildeHDR)

    I_TM = (I_tildeHDR * (1 + (I_tildeHDR / (I_white ** 2)))) / (1 + I_tildeHDR)
    return I_TM


####### PHOTOGRAPHIC TONEMAPPING #######

####### CREATE AND TONEMAP YOUR OWN HDR PHOTO #######

####### NOISE CALIBRATION #######

####### MERGING WITH OPTIMAL WEIGHTS #######

def main():

    # colorchecker = io.imread('assgn2/data/door_stack/exposure15.jpg')
    # # print(colorchecker.shape)
    # plt.imshow(colorchecker)
    # x = plt.ginput(8)
    # print(x)

    # plt.show()

    print("Initializing variables...")
    N = 1
    width = 0
    height = 0

    tuple_im = tuple()
    t = np.zeros(16)
    for k in range(16):
        im = io.imread('assgn2/data/door_stack/exposure'+str(k+1)+'.jpg')[::N, ::N]
        im_flattened = im.reshape((im.shape[0] * im.shape[1], im.shape[2]))
        tuple_im += (im_flattened,)
        (height, width, _) = im.shape

        t_k = (2**k) / 2048
        t[k] = t_k

    jpg = np.stack(tuple_im, axis=-1)

    downsampled_squares = (colorchecker_squares / N).astype(int)

    print("Finished initialization!")

    I_lin = linearize_rendered_images(jpg, t)
    I_HDR = merge_exposure_stack(jpg, I_lin, t, width, height, 'lin')
    # colorchecker_averages(I_HDR, downsampled_squares, width, height)
    


if __name__ == "__main__":
    main()





                       
                    #    (, ), (, ),
                    #    (, ), (, ),
                    #    (, ), (, ),
                    #    (, ), (, ),]

# row1 = [(3314.0161290322576, 1420.0645161290327), (3422.887096774193, 1541.0322580645166), (3477.322580645161, 1426.1129032258068), (3574.0967741935474, 1528.9354838709683), (3628.532258064515, 1420.0645161290327), (3737.403225806451, 1534.9838709677424), (3773.6935483870966, 1420.0645161290327), (3888.612903225806, 1522.887096774194)]
# row2 = [(3307.967741935483, 1262.8064516129039), (3410.790322580645, 1365.6290322580649), (3477.322580645161, 1268.854838709678), (3574.0967741935474, 1371.677419354839), (3622.4838709677415, 1262.8064516129039), (3719.258064516129, 1365.6290322580649), (3779.7419354838703, 1262.8064516129039), (3876.5161290322576, 1359.5806451612907)]
# row3 = [(3307.967741935483, 1111.5967741935488), (3416.8387096774186, 1220.4677419354844), (3459.177419354838, 1105.5483870967746), (3561.999999999999, 1208.370967741936), (3622.4838709677415, 1105.5483870967746), (3719.258064516129, 1202.322580645162), (3779.7419354838703, 1105.5483870967746), (3864.4193548387093, 1202.322580645162)]
# row4 = [(3295.8709677419347, 948.2903225806458), (3398.6935483870966, 1045.0645161290327), (3453.1290322580635, 942.2419354838717), (3555.9516129032254, 1051.1129032258068), (3610.387096774193, 942.2419354838717), (3701.112903225806, 1045.0645161290327), (3767.645161290322, 954.33870967742), (3858.3709677419347, 1051.1129032258068)]
# row5 = [(3289.822580645161, 784.9838709677424), (3392.645161290322, 887.8064516129039), (3453.1290322580635, 784.9838709677424), (3549.903225806451, 875.7096774193551), (3604.3387096774186, 778.9354838709683), (3701.112903225806, 887.8064516129039), (3767.645161290322, 797.0806451612907), (3858.3709677419347, 887.8064516129039)]
# row6 = [(3289.822580645161, 627.7258064516136), (3362.403225806451, 712.4032258064517), (3453.1290322580635, 627.7258064516136), (3525.709677419354, 706.354838709678), (3610.387096774193, 633.7741935483873), (3682.967741935483, 706.354838709678), (3761.5967741935474, 633.7741935483873), (3834.177419354838, 718.4516129032263)]
