from assignment3.lib.file_system import load_dataset
from assignment3.lib.utils import center_data_frame, get_data_frame_means
import pandas as pd
import numpy as np

if __name__ == '__main__':
    dataset = load_dataset('./sound.csv')
    means = get_data_frame_means(dataset)
    centered_dataset = center_data_frame(dataset, means)
    covariance_matrix = pd.DataFrame(centered_dataset).cov()
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    transformed_dataset = np.array(eigen_vectors).T.dot(np.array(centered_dataset).T)
    print(transformed_dataset.T)
