import matplotlib.pyplot as plt
import numpy as np
import math
from skimage import io
import getopt, sys

from cp_hw2 import writeHDR, read_colorchecker_gm, lRGB2XYZ, XYZ2lRGB

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
    Z = jpg

    # Initialize known matrices
    n = 256
    A = np.zeros((Z.size + n, n + Z.shape[0]))
    b = np.zeros(A.shape[0])

    w = w_uniform

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

    I_lin = np.exp(vector_g[Z])
    print(I_lin)

    np.save('g_jpg_uniform.npy', vector_g)

    return I_lin


####### MERGING EXPOSURE STACK INTO HDR IMAGE #######
def merge_exposure_stack(I_LDR, I_lin, t, width, height, merge='lin'):
    w = w_uniform
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

    writeHDR('uniform' + merge + '.HDR', I_HDR)
    
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
    for i in range(colorchecker_coords.shape[0]):
        (x_min, y_min, x_max, y_max) = colorchecker_coords[i]
        square = I_HDR[y_min:y_max, x_min:x_max, :]
        r_mean = np.mean(square[:,:,0])
        g_mean = np.mean(square[:,:,1])
        b_mean = np.mean(square[:,:,2])
        for j in range(3):
            A[3*i+j][4*j] = r_mean
            A[3*i+j][4*j+1] = g_mean
            A[3*i+j][4*j+2] = b_mean
            A[3*i+j][4*j+3] = 1
        b[3*i] = red[i % 4, i // 4]
        b[3*i+1] = green[i % 4, i // 4]
        b[3*i+2] = blue[i % 4, i // 4]

    rgb_transform = np.linalg.lstsq(A, b, rcond=None)[0].reshape(3,4)

    homogenous_hdr = np.insert(I_HDR, 3, 1, axis=2)

    flattened_hdr = homogenous_hdr.reshape(height * width, 4)

    new_hdr = np.matmul(rgb_transform, flattened_hdr.T).T
    new_hdr = np.reshape(new_hdr, (height, width, 3))
    new_hdr = np.clip(new_hdr, 0, 100)

    writeHDR('colorcorrected.HDR', new_hdr)
    return new_hdr


def white_balancing(I_HDR, colorchecker_coords, width, height):
    (x_min, y_min, x_max, y_max) = colorchecker_coords[3] # white is 4th square
    white_square = I_HDR[y_min:y_max, x_min:x_max, :]
    r_mean = np.mean(white_square[:,:,0])
    g_mean = np.mean(white_square[:,:,1])
    b_mean = np.mean(white_square[:,:,2])

    scale = np.array([[1 / r_mean, 0, 0],
                      [0, 1 / g_mean, 0],
                      [0, 0, 1 / b_mean]])

    flattened_hdr = I_HDR.reshape(height * width, 3)

    new_hdr = np.matmul(scale, flattened_hdr.T).T
    new_hdr = np.reshape(new_hdr, (height, width, 3))
    new_hdr = np.clip(new_hdr, 0, 100)

    writeHDR('whitebalanced.HDR', new_hdr)

    return new_hdr


####### PHOTOGRAPHIC TONEMAPPING #######
def tonemap_channel(hdr_channel):
    N = hdr_channel.size
    K = 0.15
    B = 0.95
    epsilon = 0.01

    I_mHDR = np.exp((1/N) * np.sum(np.log(hdr_channel + epsilon)))
    I_tildeHDR = K * (hdr_channel / I_mHDR)
    I_white = B * np.max(I_tildeHDR)

    I_TM = (I_tildeHDR * (1 + (I_tildeHDR / (I_white ** 2)))) / (1 + I_tildeHDR)
    return I_TM

def tonemapping(I_HDR, method='rgb'):
    if (method == 'rgb'):
        R_TM = tonemap_channel(I_HDR[:,:,0])
        G_TM = tonemap_channel(I_HDR[:,:,1])
        B_TM = tonemap_channel(I_HDR[:,:,2])
        I_TM = np.dstack((R_TM, G_TM, B_TM))

    else: # method == 'lum' (luminance)
        XYZ = lRGB2XYZ(I_HDR)
        X = XYZ[:,:,0]
        Y = XYZ[:,:,1]
        Z = XYZ[:,:,2]

        x = np.nan_to_num(X / (X + Y + Z))
        y = np.nan_to_num(Y / (X + Y + Z))
        Y = tonemap_channel(Y)

        X_TM = np.nan_to_num(Y * x / y)
        Y_TM = Y
        Z_TM = np.nan_to_num(Y * ( 1 - x - y ) / y)
        I_TM = XYZ2lRGB(np.dstack((X_TM, Y_TM, Z_TM)))

    writeHDR('tonemapped_' + method + '.HDR', I_TM)
    return I_TM



def help():
    print("wrong inputs")

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:i:m:w:t:", ["help", "downsample=", "image=", "merge=", "weight=", "tonemap="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)
        help()
        sys.exit(2)

    N = 8
    image = 'jpg'
    merge = 'lin'
    weight = 'uniform'
    tonemap = 'rgb'
    for o, a in opts:
        if o == "-h":
            help()
            sys.exit()
        elif o in ("-d", "--downsample"):
            N = int(a)
        elif o in ("-i", "--image"):
            image = a
        elif o in ("-m", "--merge"):
            merge = a
        elif o in ("-w", "--weight"):
            weight = a
        elif o in ("-t", "--tonemap"):
            tonemap = a
        else:
            assert False, "unhandled option"
            
    print("Initializing variables...")
    stack = 16
    width = 0
    height = 0

    tuple_im = tuple()
    t = np.zeros(stack)
    for k in range(stack):
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
    I_HDR = merge_exposure_stack(jpg, I_lin, t, width, height, merge)
    I_HDR = colorchecker_averages(I_HDR, downsampled_squares, width, height)
    I_HDR = white_balancing(I_HDR, downsampled_squares, width, height)
    I_HDR = tonemapping(I_HDR, tonemap)
    

if __name__ == "__main__":
    main()