from typing import Optional
import numpy as np

class MatrixProcessor:
    """Class for handling matrix-related operations such as calculating means and concatenating matrices."""
    
    def __init__(self, matrix: list[np.ndarray], P: Optional[np.ndarray] = None):
        """
        Initializes the MatrixProcessor with the given matrix and optional parameter P.
        
        Args:
            matrix (list of ndarray): A list of numpy arrays to be processed.
            P (ndarray, optional): A transformation matrix for the concatenation step.
        """
        self.matrix = matrix
        self.P = P

    def calculate_means(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the means of specific slices of the matrix assuming a specific shape.

        The matrix must have shape (1, 4, 2), and this function computes the means 
        for two slices: one for the first four elements and one for the last four elements.
        
        Returns:
            tuple: The mean of the first slice (cx) and the second slice (cy).
        
        Raises:
            ValueError: If the shape of the matrix is not (1, 4, 2).
        """
        a = self.matrix[0]
        if a.shape != (1, 4, 2):
            raise ValueError("The array should have shape (1, 4, 2).")

        cx = np.mean(a[0, :4], axis=0)
        cy = np.mean(a[0, -4:], axis=0)
        
        return cx, cy

    def filter_non_empty_matrices(self) -> tuple[list[np.ndarray], list[int]]:
        """
        Filters out empty matrices from the list and returns the non-empty matrices 
        along with their corresponding indices.
        
        Returns:
            tuple: A list of non-empty matrices and their indices in the original matrix.
        """
        non_empty_indices = [index for index, mat in enumerate(self.matrix) if mat[0].size > 0]
        non_empty_matrix = [m[0] for m in self.matrix if m[0].size > 0]
        return non_empty_matrix, non_empty_indices

    def verify_matrices(self, non_empty_matrix) -> bool:
        """
        Verifies if there are enough non-empty matrices to proceed with concatenation.
        
        Args:
            non_empty_matrix (list of ndarray): List of non-empty matrices to verify.
        
        Returns:
            bool: True if there are at least two non-empty matrices, False otherwise.
        """
        return len(non_empty_matrix) >= 2

    def concatenate_matrices(self, non_empty_matrix, non_empty_indices) -> np.ndarray:
        """
        Concatenates the non-empty matrices into a single matrix based on specific transformation criteria.
        
        Args:
            non_empty_matrix (list of ndarray): List of non-empty matrices to concatenate.
            non_empty_indices (list of int): Indices of the non-empty matrices in the original list.
        
        Returns:
            ndarray: The concatenated matrix.
        """
        lines = []
        len_matrix = len(non_empty_matrix)

        for i in range(len_matrix):
            reshaped_matrix = -non_empty_matrix[i].reshape(-1, 1)
            reshaped_matrix = np.vstack((reshaped_matrix, -np.ones((1, 1))))

            # Constructing the line to append to the result
            line = np.hstack((
                self.P[non_empty_indices[i]], 
                np.zeros((reshaped_matrix.shape[0], i)), 
                reshaped_matrix, 
                np.zeros((reshaped_matrix.shape[0], len_matrix - i - 1))
            ))

            lines.append(line)

        matrix_result = np.vstack(lines)
        return matrix_result

    def process_and_concatenate(self) -> Optional[np.ndarray]:
        """
        Filters non-empty matrices, verifies if there are enough matrices for concatenation, 
        and then concatenates them.
        
        Returns:
            ndarray or None: The concatenated matrix or None if there are not enough non-empty matrices.
        """
        non_empty_matrix, non_empty_indices = self.filter_non_empty_matrices()

        if not self.verify_matrices(non_empty_matrix):
            return None

        matrix_result = self.concatenate_matrices(non_empty_matrix, non_empty_indices)
        return matrix_result
