__author__ = 'mac'

from scipy.cluster.hierarchy import linkage, leaves_list, cophenet, dendrogram
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
import numpy as np
import scipy.stats
from tqdm import tqdm
import pathos.multiprocessing as mp
from multiprocessing import Array
import os
import ctypes


def reorder_connectivity(connmat,get_orginial_indexing=False):
    connmat.astype(np.float64)
    Y = 1 - connmat
    # print 'Y.shape: ',Y.shape
    # print 'squareform(Y).shape: ',squareform(Y).shape
    # print 'squarform(Y): \n',squareform(Y)
    # print connmat.min()
    # print connmat.max()
    # print Y.min()
    # print Y.max()
    Z = linkage(squareform(Y,checks=False),method='average')
    indexing = leaves_list(Z)
    indexing = indexing[::-1]#reverse it because we did 1-connmat
    outmat = connmat[:,indexing][indexing,:]
    if(get_orginial_indexing):
        return (outmat, indexing)
    else:
        return outmat


def cluster_correlation_matrix(correlation_matrix, get_original_indexing=False,get_reordered_matrix=False, meth='average'):
    Y=1-correlation_matrix
    link, original_idx = UPGMACluster(Y,get_original_indexing=True,meth=meth)
    original_idx = original_idx
    reordered = correlation_matrix[:,original_idx][original_idx,:]

    if(get_original_indexing and get_reordered_matrix):
        return link, original_idx, reordered
    elif(get_original_indexing):
        return link, original_idx
    elif(get_reordered_matrix):
        return link, reordered
    else:
        return link




def UPGMACluster(distancemat,get_original_indexing=False,get_reorderd_matrix=False,meth='average'):
    Y = distancemat
    if(distancemat.shape[0] == distancemat.shape[1]):
        Z = linkage(squareform(Y,checks=False),method=meth)

    else:
        Z = linkage(squareform(Y),method=meth)

    # dend = dendrogram(Z,no_plot=True)
    indexing = leaves_list(Z)
    # indexing = dend['leaves']
    # indexing = indexing[::-1]
    outmat = distancemat[:,indexing][indexing,:]
    if(get_original_indexing and get_reorderd_matrix):
        return Z, indexing, outmat
    elif(get_reorderd_matrix):
        return Z, outmat
    elif(get_original_indexing):
        return Z, indexing
    else:
        return Z


def calculate_connectivity(inputdata,metaaaxis=0):
    """
    Computes the connectivity matrix for inputdata given a kxm matrix with k rows representing
    k possible classes (the reverse row/column case can be computed by resetting the metaaxis variable).
    This is done by seeing for each column, which among k rows contains the largest value. If two columns
    i and j both have their greatest value in the gth row (of k possible rows), then in output matrix O, O(i,j) = 1.0
    """

    if(not isinstance(inputdata,np.ndarray)):raise TypeError('Input matrix must be of type numpy.ndarray.')

    if(metaaaxis==0):axis =1
    else:axis=0

    nClusters = inputdata.shape[metaaaxis]
    nSamples = inputdata.shape[axis]

    idxs = np.argmax(inputdata,axis=metaaaxis)
    idxs.shape = (len(idxs),1)

    conn = np.zeros((nSamples,nSamples),dtype='float')

    conn[idxs == idxs.T] = 1.0

    return conn


def calculate_cophenetic_correlation(connmat):
    Y = 1 - connmat
    Z = linkage(squareform(Y),method='average')
    c,d= cophenet(Z,squareform(Y))
    #print c
    #print d
    return (c,d)


def _init_pool(matrix, shared_r_arr_, shared_p_arr_, nRows, symmetrical):
    global shared_matrix
    shared_matrix = matrix
    global shared_nRows
    shared_nRows = nRows
    global shared_symmetrical
    shared_symmetrical = symmetrical
    global shared_r_arr
    shared_r_arr = shared_r_arr_
    global shared_p_arr
    shared_p_arr = shared_p_arr_


def _f(i):
    array_r = np.frombuffer(shared_r_arr.get_obj())
    array_r = array_r.reshape((shared_nRows, shared_nRows))
    array_p = np.frombuffer(shared_p_arr.get_obj())
    array_p = array_p.reshape((shared_nRows, shared_nRows))
    if shared_symmetrical:
        for j in range(i, shared_nRows):
            rowidata = shared_matrix[i, :]
            rowjdata = shared_matrix[j, :]
            r, p = scipy.stats.pearsonr(rowidata, rowjdata)
            array_r[i, j] = array_r[j, i] = r
            array_p[i, j] = array_p[j, i] = p
            # with shared_r_arr.get_lock():
            #     array_r[i, j] = array_r[j, i] = r
            # with shared_p_arr.get_lock():
            #     array_p[i, j] = array_p[j, i] = p
    else:
        for j in range(0, shared_nRows):
            rowidata = shared_matrix[i, :]
            rowjdata = shared_matrix[j, :]
            r, p = scipy.stats.pearsonr(rowidata, rowjdata)
            with shared_r_arr.get_lock():
                array_r[i, j] = r
            with shared_p_arr.get_lock():
                array_p[i, j] = p
    return i


def calculatePearsonCorrelationMatrixMultiprocessing(matrix, axis=0, symmetrical=True, getpvalmat=False):

    if axis == 1:
        matrix = matrix.T

    nRows = matrix.shape[0]

    # create shared array that can be used from multiple processes
    output_r_arr = Array(ctypes.c_double, matrix.shape[0] * matrix.shape[0])
    # then in each new process create a new numpy array using:
    output_r = np.frombuffer(output_r_arr.get_obj())  # mp_arr and arr share the same memory
    # make it two-dimensional
    output_r = output_r.reshape((matrix.shape[0], matrix.shape[0]))  # b and arr share the same memory
    # output_r = np.zeros((nRows,nRows))  # old version

    output_p_arr = Array(ctypes.c_double, matrix.shape[0] * matrix.shape[0])
    output_p = np.frombuffer(output_p_arr.get_obj())
    output_p = output_p.reshape((matrix.shape[0], matrix.shape[0]))

    print 'Calculating Pearson R for each row, multithreaded'
    print mp.cpu_count(), 'processes in pool'

    pool = None
    try:
        pool = mp.Pool(mp.cpu_count(),
                       initializer=_init_pool,
                       initargs=(matrix, output_r_arr, output_p_arr,
                                 nRows, symmetrical))

        # bar = tqdm(total=nRows*nRows/2)
        # tqdm.write('Calculating Pearson R for each row, multithreaded')
        for result in tqdm(pool.imap_unordered(_f, range(0, nRows)), total=nRows):
            # bar.update(result)
            pass
        # bar.close()
    finally:  # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    print output_r

    if getpvalmat:
        return output_r, output_p
    else:
        return output_r


def calculatePearsonCorrelationMatrix(matrix, axis=0, symmetrical=True, getpvalmat=False, verbose=False):

    if axis == 1:
        matrix = matrix.T

    nRows = matrix.shape[0]

    output_r = np.zeros((nRows, nRows))
    output_p = np.zeros((nRows, nRows))


    if verbose:
        bar = tqdm(total=nRows)
        tqdm.write('Calculating Pearson R for each row')
    for i in range(0, nRows):
        if symmetrical:
            for j in range(i, nRows):
                rowidata = matrix[i,:]
                rowjdata = matrix[j,:]
                r,p = scipy.stats.pearsonr(rowidata,rowjdata)
                output_r[i,j] = output_r[j,i] = r
                output_p[i,j] = output_p[j,i] = p
        else:
            for j in range(0, nRows):
                rowidata = matrix[i,:]
                rowjdata = matrix[j,:]
                r,p = scipy.stats.pearsonr(rowidata,rowjdata)
                output_p[i,j] = p
                output_r[i,j] = r
        if verbose:
            bar.update(1)

    if verbose:
        bar.close()

    if getpvalmat:
        return output_r, output_p
    else:
        return output_r


def calculateSpearmanRankCorrelationMatrix(matrix, axis=0, getpvalmat=False):

    if axis == 0:
        matrix = matrix.T

    rho, pvals = scipy.stats.spearmanr(matrix,axis=0)

    if(getpvalmat):
        return rho, pvals
    else:
        return rho




def calculateDistanceMatrix(observations,axis=0, metric='euclidean'):
    """
    X is a mxn matrix of m observations in n dimensions. axis = 0 is default,
    set axis=1 to reverse the rows and columns of X
    """
    if(axis==1):
        observations = observations.T

    Y = pdist(observations,metric)
    return squareform(Y)




