

# from config import setup_argparse
import numpy as np
import tensorflow as tf
import os

#############################################################################
####                               Read data                            #####
#############################################################################
### read for classification purpose
### last column as label (0 or 1), other are features
### for "spambase" we need an extra preprocess step
def read_dataset_file_my(dataset, attackedClass, attackingClass, test_start_index=0, is_train=None):
    dim12 = 28
    dim3 = 1
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        dim12 = 32
        dim3 = 3
    else:
        raise TypeError("Error: Dataset {} does not exist!".format(dataset))

    x = np.concatenate((x_train, x_test), axis=0)
    x = np.true_divide(x,255)
    y = np.concatenate((y_train, y_test), axis=0)
    x,y = mnist_preprocess_data(x, y, attackedClass, attackingClass) #shape: (4601, 54)
    x = np.array(x)
    x = x.reshape(-1,dim12*dim12*dim3)

    if is_train == None:
        return np.matrix(x), y

    if is_train == True:
        x = x[:test_start_index]
        y = y[:test_start_index]
    elif is_train == False:
        x = x[test_start_index:]
        y = y[test_start_index:]
    return np.matrix(x), y

### ------------------------------------------------------------------------------
### converts one-hot y to a single numberic value
### only reads 1 and 7. label(1):-1 label(7):1
def mnist_preprocess_data(x,y, attackedClass, attackingClass):
    x_tmp = [xi for xi,yi in zip(x,y) if yi==attackedClass or yi==attackingClass]
    y_tmp = [yi for xi,yi in zip(x,y) if yi==attackedClass or yi==attackingClass]
    y_tmp = [0 if yi==attackedClass else 1 for yi in y_tmp]
    return x_tmp, y_tmp


# -------------------------------------------------------------------------------
### Select a certain points for train,test and outlier so we have same data for Cross validation
### Should be executed only once (at ONE run of the code)
### then we should read indices from saved file by calling sample_CV_dataset_ReadfromFile_my()
def sample_CV_dataset_WriteinFile_my(trnct, tstct, vldct, seed, filename , y):

    f = open(filename, "w+")
    for i in range(10):
        seed_itr = seed + i
        sampletrn, sampletst, samplevld = sample_dataset_my(trnct, tstct, vldct, seed_itr, y)
        f.write(','.join([str(val) for val in sampletrn]) + '\n')
        f.write(','.join([str(val) for val in sampletst]) + '\n')
        f.write(','.join([str(val) for val in samplevld]) + '\n')

    f.close()

# -------------------------------------------------------------------------------
### select a certain points for train,test and outlier so we have same data for Cross validation
### we should read indices from saved file
### itr_num: number of the iteration of the program, it depends on us until what iteration we want read data
### in each iteration we should call the function with the corresponding itr_num again (MAX is 20)
### x,y: all initial dataset without any change in indices (but data may just preprocessed)
def sample_CV_dataset_ReadfromFile_my(x, y, itr_num, filename):

    with open(filename) as f:
        # go forward into the file to reach the saved itr_num indices
        for i in range(itr_num):
            for j in range(3):
                f.readline()

        sampletrn = list(map(int, f.readline().split(',')))
        trnx = np.matrix([np.array(x[row]).reshape((x.shape[1],)) for row in sampletrn])
        trny = [y[row] for row in sampletrn]

        sampletst = list(map(int, f.readline().split(',')))
        tstx = np.matrix([np.array(x[row]).reshape((x.shape[1],)) for row in sampletst])
        tsty = [y[row] for row in sampletst]

        samplevld = list(map(int, f.readline().split(',')))
        vldx = np.matrix([np.array(x[row]).reshape((x.shape[1],)) for row in samplevld])
        vldy = [y[row] for row in samplevld]


        return trnx, trny, tstx, tsty, vldx, vldy

# -------------------------------------------------------------------------------
### added training point for outlier detection (exact number is read from args)
### restricted the way of selecting outlier points; from all over x to only training points
def sample_dataset_my(trnct, tstct, vldct, seed, y):

    # to have two balances classes
    ind_1 = [i for i,yi in enumerate(y) if yi==0]
    ind_2 = [i for i,yi in enumerate(y) if yi==1]
    np.random.seed(seed)
    fullperm1 = np.random.permutation(ind_1)
    fullperm2 = np.random.permutation(ind_2)

    trnct,tstct,vldct  = trnct//2,tstct//2,vldct//2

    sampletrn,sampletst,samplevld = [],[],[]
    sampletrn.extend(fullperm1[:trnct])
    sampletrn.extend(fullperm2[:trnct])

    sampletst.extend(fullperm1[trnct:trnct + tstct])
    sampletst.extend(fullperm2[trnct:trnct + tstct])

    samplevld.extend(fullperm1[trnct + tstct:trnct + tstct + vldct])
    samplevld.extend(fullperm2[trnct + tstct:trnct + tstct + vldct])

    if ((trnct + tstct + vldct ) <= len(fullperm1)) and ((trnct + tstct + vldct) <= len(fullperm2)):
        pass
    else:
        print("Error! Not enough data! [in sample_dataset_my()]")
        exit(1)

    return sampletrn, sampletst, samplevld


def join_attacks(test_start_index, num_train_runs, num_runs, indexFile, partialIndexFolder):

    for itr_num in range(10):
        sampletrn,sampletst,samplevld,sampleoutl,sampleother = [],[],[],[],[]

        ### train part indices
        for file_num in range(num_runs):

            ### indicate train or test part indices
            scale = 0 if file_num < num_train_runs else test_start_index

            indexFile1 = os.path.join(partialIndexFolder, "indexCV_mnist_SVM"+str(file_num)+".txt")
            with open(indexFile1) as f:
                for i in range(itr_num):
                    for j in range(3):
                        f.readline()

                tmp = list(map(int, f.readline().split(',')))
                tmp = [ind + scale for ind in tmp]
                sampletrn += tmp
                tmp = list(map(int, f.readline().split(',')))
                tmp = [ind + scale for ind in tmp]
                sampletst += tmp
                tmp = list(map(int, f.readline().split(',')))
                tmp = [ind + scale for ind in tmp]
                samplevld += tmp
                line = f.readline().split(',')


        with open(indexFile, "a") as f:
            f.write(','.join([str(val) for val in sampletrn]) + '\n')
            f.write(','.join([str(val) for val in sampletst]) + '\n')
            f.write(','.join([str(val) for val in samplevld]) + '\n')

    return

