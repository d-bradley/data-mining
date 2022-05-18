import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


def read_txt():
    with open('values.txt', 'r') as f:
        matrix = f.read()
        matrix = [item.split() for item in matrix.split('\n')]
    return matrix


def center(matrix):

    count = 0
    total = 0
    mean_array = []

    for count in range(len(matrix[0])):
        for i in range(len(matrix)):
            matrix[i][count] = float(matrix[i][count])
            colValue = matrix[i][count]
            total = colValue + total
            mean = total/len(matrix)
        mean_array.append(mean)
        total = 0
        count = count + 1

    mean_array = np.array(mean_array)
    matrix_array = np.array(matrix)
    center_matrix = np.array(matrix_array)

    for index, mean in enumerate(mean_array):
        center_matrix[:,index] -= mean

    return center_matrix


def covar(centered_matrix):
    count = 0
    centered_transpose = np.transpose(centered_matrix)
    cov_dim = len(centered_transpose)
    num_rows = len(centered_matrix)
    cov_matrix = np.zeros([cov_dim, cov_dim])

    for i in range(cov_dim):
        for j in range(cov_dim):
            xi = np.expand_dims(centered_matrix[:,i],axis=(0))
            xj = np.expand_dims(centered_matrix[:,j],axis=(0))
            xj_t = xj.transpose()
            cov = np.dot(xi,xj_t)/num_rows
            cov_matrix[i,j] = cov
    return cov_matrix


def eigen(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    return eigenvalues, eigenvectors


def minimum_pc(eigenvalues, alpha):
    total = sum(eigenvalues)
    cumulative = 0
    for i, eigval in enumerate(eigenvalues):
        cumulative += eigval
        if cumulative / total > alpha:
            print(f"{i+1} principal components needed.")
            return i + 1
    return 0


def plot_components(eigvects):
    plt.scatter(range(1,len(eigvects[0])+1), eigvects[0], label='first component')
    plt.scatter(range(1,len(eigvects[1])+1), eigvects[1], label='second component')
    plt.xlabel('Dimensions')
    plt.ylabel('Magnitude')
    plt.title('Magnitude of 1st and 2nd Components with respect to Dimension')
    plt.legend(loc='upper left')
    plt.show()


def save_components_to_text(eigvects):
    components = eigvects[0:2]
    np.savetxt("Components.txt", components, delimiter=",")


def get_reduced(centered_matrix,eig_vectors):
    eig_vects = eig_vectors.transpose()
    eig_vects = eig_vects[:, :2]
    projected = centered_matrix.dot(eig_vects)
    return projected


def plot_reduced(projected):
    plt.scatter(projected[:,0],projected[:,1])
    plt.xlabel('a1')
    plt.ylabel('a2')
    plt.show()


def test_sklearn_PCA(matrix):
    pca = PCA()
    pca.fit(matrix)
    print(pca.components_)


if __name__ == '__main__':
    orig_matrix = read_txt()
    cov_matrix = covar(center(orig_matrix))
    eigvals, eigvects = eigen(cov_matrix)
    min_pcs = minimum_pc(eigvals, 0.9)
    plot_components(eigvects)
    save_components_to_text(eigvects)
    projected = get_reduced(center(orig_matrix), eigvects)
    plot_reduced(projected)
    test_sklearn_PCA(orig_matrix)

    # the retained variance of r=2 is 99.69%.
    pass