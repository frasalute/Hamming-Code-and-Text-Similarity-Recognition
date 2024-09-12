import numbers

class snp:

    def is_valid_array(array: list) -> bool:
        """
        Checks if the array is valid as a matrix or vector, if not it raises an exception.
        """
        if not isinstance(array, list):
            raise Exception("Not valid array: Not a list")

        if array == []:
            raise Exception("Not valid vector: Empty")
        
        # Checks if there are any non-number elements in the vector.
        if isinstance(array[0], numbers.Number):
            if not all(isinstance(element, numbers.Number) for element in array):
                raise Exception("Not valid vector: Non-numbers  entry")
            return True
        
        if array[0] == []:
            raise Exception("Not valid matrix: Empty")
        
        # Checks if there are any non-number elements in the matrix and if every row has the same columns.
        columns = len(array[0])
        for row in array:
            if len(row) != columns:
                raise Exception("Number of columns not consistent accross aug_matrix")
            
            if not all(isinstance(element, numbers.Number) for element in row):
                raise Exception(f"Not valid array: Non-number entry in row {row}")
            
        return True

    def is_vector(array: list) -> bool:
        """
        Returns true if the array is a vector.
        """
        return isinstance(array[0], numbers.Number)
    
    def ones(length: int) -> list:
        """
        Returns an array of ones with the desired length.
        """
        if not isinstance(length, int):
            raise Exception("Not a valid number")
        
        array = [1 for i in range(length)]
        return array
    
    def zeros(length: int) -> list:
        """
        Returns an array of zeros with the desired length.
        """
        if not isinstance(length, int):
            raise Exception("Not a valid number")
        
        array = [0 for i in range(length)]
        return array
    
    def reshape(array: list, dimensions: tuple) -> list:
        """
        Reshapes an array into a new matrix/vector with the desired dimensions.
        """
        snp.is_valid_array(array)

        if not isinstance(dimensions, tuple):
            raise Exception("Not a valid tuple")
        if len(dimensions) != 2:
            raise Exception("Not valid tuple dimensions")
        if not isinstance(dimensions[0], int) or not isinstance(dimensions[1], int):
            raise Exception("Non int entries in the tuple")

        if snp.shape(array)[0] * snp.shape(array)[1] != dimensions[0] * dimensions[1]:
            raise Exception("Reshape not possible, number of elements not equal to rows * cols of dimensions")
        
        #If the array is a matrix, converts the entries of a matrix into a vector.
        if not snp.is_vector(array):
            vector = []
            for row in array:
                vector += row
            array = vector
        
        #Returns a vector if dimension is (1, number)
        if dimensions[0] == 1:
            return array

        number_of_rows = dimensions[0]
        number_of_columns = dimensions[1]
        new_array = []
        index = 0

        #Places every entry of the array in the new matrix/vector in the correct position, row by row.
        for i in range(number_of_rows):
            new_row = []
            for j in range(number_of_columns):
                new_row.append(array[j+index])
            index += number_of_columns
            new_array.append(new_row)
        
        return new_array
    
    def shape(array: list) -> tuple:
        """
        Returns the shape of an array.
        """
        snp.is_valid_array(array)

        if not snp.is_vector(array): #If matrix
            return (len(array), len(array[0]))
        else: #If vector
            return (1, len(array))
    
    def append(array1: list, array2: list) -> list:
        """
        Adds the elements of the two arrays together.
        """
        snp.is_valid_array(array1)
        snp.is_valid_array(array2)
        
        if isinstance(array1[0], list) and isinstance(array2[0], numbers.Number):
            raise Exception("Error: Array 1 is a matrix while Array 2 a vector")
        
        if isinstance(array1[0], numbers.Number) and isinstance(array2[0], list):
            raise Exception("Error: Array 2 is a matrix while Array 1 a vector")
        
        if isinstance(array1[0], list) and isinstance(array2[0], list):
            if snp.shape(array1)[1] != snp.shape(array2)[1]:
                raise Exception("Error: Array 2 and Array 1 have a different number of columns")
        
        return array1 + array2
    
    def get(array:list, position: tuple) -> numbers.Number:
        """
        Returns the element at the given position in the array.
        """
        snp.is_valid_array(array)

        if not isinstance(position, tuple):
            raise Exception("Not a valid tuple")
        if len(position) != 2:
            raise Exception("Not valid tuple dimensions")
        if not isinstance(position[0], int) or not isinstance(position[1], int):
            raise Exception("Non int entries in the tuple")

        try:
            if snp.is_vector(array) and position[0] == 0: #If vector
                return array[position[1]]
            else: #If matrix
                return array[position[0]][position[1]]
        except:
            raise Exception("Invalid position in the array")

    def add(array1: list, array2: list) -> list:
        """
        Adds the elements of the two vectors/matrices together, element by element.
        """
        snp.is_valid_array(array1)
        snp.is_valid_array(array2)

        if snp.shape(array1) != snp.shape(array2):
            raise Exception("Matrices/Vectors have a shape mismatch.")

        if snp.is_vector(array1):
            new_array = []
            for i, j in zip(array1, array2):
                new_array.append(i + j)
            return new_array
        
        new_array = []
        #If the arrays are matrices, sum the elements row by row and append it to the new array.
        for row_index in range(len(array1)):
            new_row = []
            for i, j in zip(array1[row_index], array2[row_index]):
                new_row.append(i + j)
            new_array.append(new_row)
        return new_array
    
    def substract(array1: list, array2: list) -> list:
        """
        Substracts the elements of the two vectors/matrices together, element by element.
        """
        snp.is_valid_array(array1)
        snp.is_valid_array(array2)

        if snp.shape(array1) != snp.shape(array2):
            raise Exception("Matrices/Vectors have a shape mismatch.")
        
        if snp.is_vector(array1):
            new_array = []
            for i, j in zip(array1, array2):
                new_array.append(i - j)
            return new_array
        
        new_array = []
        #If the arrays are matrices, substracts the elements row by row and append it to the new array.
        for row_index in range(len(array1)):
            new_row = []
            for i, j in zip(array1[row_index], array2[row_index]):
                new_row.append(i - j)
            new_array.append(new_row)
        return new_array
    
    def dot_product(array1: list, array2: list) -> list:
        """
        Performs the dot product between two arrays.
        """
        snp.is_valid_array(array1)
        snp.is_valid_array(array2)

        #Dot product between two vectors.
        if snp.is_vector(array1) and snp.is_vector(array2):
            if len(array1)==len(array2):
                new_array = []
                for i, j in zip(array1, array2):
                    new_array.append(i * j)
                return [sum(new_array)]
            else:
                raise Exception("Dot Product can't be calculated, vectors have different lengths.")

        if snp.shape(array1)[1] != snp.shape(array2)[0]:
            raise Exception("Dot Product can't be calculated, shape mismatch.")
        
        new_array = []
        if snp.is_vector(array1): #If the array_1 is vector and the array_2 a matrix
            for col in range(len(array2[0])):
                new_row = []
                for row in range(len(array1)):
                    new_row.append(array1[row]*array2[row][col])
                new_array.append(sum(new_row))

        else: #If both arrays are matrices
            for row_index in range(len(array1)):
                for col_index in range(len(array2[0])):
                    col = []
                    for index in range(len(array2)):
                        col.append(array1[row_index][index] * array2[index][col_index])
                    new_array.append(sum(col))

        return snp.reshape(new_array,(snp.shape(array1)[0],snp.shape(array2)[1]))
    
    def gaussian_solver(array_a: list, array_b: list) -> list:
        """
        Finds the solution to the system of equations: Ax = b, by gaussian elimination.
        """
        snp.is_valid_array(array_a)
        snp.is_valid_array(array_b)
        
        non_leading_rows = [i for i in range(len(array_a))]
        number_of_equations = len(array_a)
        result = []

        if snp.is_vector(array_a):
            raise Exception("Array_a must be a matrix")

        if not snp.is_vector(array_b):
            raise Exception("Array_b must be a vector")
        
        if snp.shape(array_a)[0] != snp.shape(array_b)[1]:
            raise Exception("Dimensions of matrices/vectors have a shape mismatch.")
        
        #Building augmented matrix
        aug_matrix = []
        for row_index, row in enumerate(array_a):
            new_row = [row + [array_b[row_index]]]
            aug_matrix += new_row

        #Performing gaussian elimination
        for col_ind in range(number_of_equations):
            for row_ind in non_leading_rows:
                if aug_matrix[row_ind][col_ind] != 0:  
                    non_leading_rows.remove(row_ind)
                    for i in range(number_of_equations):
                        if i != row_ind:
                            aug_matrix[i] = snp.substract(aug_matrix[i],[j * aug_matrix[i][col_ind]/aug_matrix[row_ind][col_ind] for j in aug_matrix[row_ind]])
                        else:
                            aug_matrix[row_ind] = [j / aug_matrix[row_ind][col_ind] for j in aug_matrix[row_ind]]
                    break
        
        #Adding the solutions to the result array
        for i in range(number_of_equations):
            if len(result) != i:
                raise Exception("Singular matrix")
            for j in range(number_of_equations):
                if aug_matrix[j][i] != 0:
                    result.append(aug_matrix[j][number_of_equations])
        
        return result


class tests:
    def test_is_valid_array():

        empty_array = []
        array1 = [1,2,3,4]
        array2 = [[1,2,3],[2,3,4],[5,6,7]]
        array3 = [[1,2,3],[2,3],[5,6,7]]
        array4 = [[1,2,3],[2,"3",4],[5,6,7]]
        array5 = [[1,2],[]]

        #Checks wether an empty array gives error.
        try:
            snp.is_valid_array(empty_array)
            raise Exception("Error")
        
        except Exception:
            pass
        
        #Checks wether a matrix with non consistent columns gives error.
        try:
            snp.is_valid_array(array3)
            raise Exception("Error")
        
        except Exception:
            pass
        
        #Checks wether non-number entries give error.
        try:
            snp.is_valid_array(array4)
            raise Exception("Error")
        
        except Exception:
            pass
        
        #Checks wether a matrix with empty columns gives error.
        try:
            snp.is_valid_array(array5)
            raise Exception("Error")
        
        except Exception:
            pass
        
        #Checks wether righ shaped arrays don't give errors.
        assert snp.is_valid_array(array1) is True
        assert snp.is_valid_array(array2) is True

    def test_ones():
        
        assert snp.ones(5) == [1,1,1,1,1]
        assert snp.ones(0) == []

        #Checks wether non-int parameters give error.
        try:
            snp.ones("hello")
            raise Exception("Error")
        except Exception:
            pass

    def test_zeros():

        assert snp.zeros(4) == [0,0,0,0]
        assert snp.zeros(0) == []

        #Checks wether non-int parameters give error.
        try:
            snp.ones("hello")
            raise Exception("Error")
        
        except Exception:
            pass
    
    def test_reshape():

        array1 = [1,2,3,4,5,6,7,8,9,10]
        array2 = [[1,2,3,4,5],[6,7,8,9,10]]

        assert snp.reshape(array1,(2,5)) == [[1,2,3,4,5],[6,7,8,9,10]]
        assert snp.reshape(array2,(5,2)) == [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]

        #Checks for shape mismatch error.
        try:
            snp.reshape(array1,(2,6))
            raise Exception("Error")
        
        except Exception:
            pass
    
    def test_shape():

        array1 = [1,2,3,4,5,6,7,8,9,10]
        array2 = [[1,2,3,4,5],[6,7,8,9,10]]
        array3 = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]

        assert snp.shape(array1) == (1,10)
        assert snp.shape(array2) == (2,5)
        assert snp.shape(array3) == (5,2)
    
    def test_append():

        array1 = [1,2,3,4,5,6,7,8,9,10]
        array2 = [1,2,3,4]
        array3 = [[1,2,3,4,5],[6,7,8,9,10]]
        array4 = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]

        assert snp.append(array1, array2) == [1,2,3,4,5,6,7,8,9,10,1,2,3,4]
        assert snp.append(array3, array3) == [[1,2,3,4,5],[6,7,8,9,10],[1,2,3,4,5],[6,7,8,9,10]]
        
        #Checks for array vector addition error.
        try:
            snp.append(array1, array3)
            raise Exception("Error")
        
        except Exception:
            pass

        #Checks for matrices shape mismatch error.
        try:
            snp.append(array3, array4)
            raise Exception("Error")
        
        except Exception:
            pass
    
    def test_get():
        array = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        array1 = [1,2,3,4,5,6,7,8,9,10]
        assert snp.get(array, (0,0)) == 1
        assert snp.get(array, (3,1)) == 8
        assert snp.get(array1, (0,1)) == 2
        assert snp.get(array1, (0,5)) == 6

        #Checks wether non-valid indices give error.
        try:
            snp.get(array1, (2,10))
            raise Exception("Error")
        
        except Exception:
            pass

    def test_add():

        array1 = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        array2 = [[1, 2], [3, 4], [5, 6]]
        array3 = [1,2,3]

        assert snp.add(array2, array2) == [[2, 4], [6, 8], [10, 12]]
        assert snp.add(array3, array3) == [2,4,6]

        #Checks wether shape mismatch between matrices gives an error.
        try:
            snp.add(array1, array2)
            raise Exception("Error")
        
        except Exception:
            pass
        
        #Checks wether addition of matrices with vectors gives an error.
        try:
            snp.add(array3, array1)
            raise Exception("Error")
        
        except Exception:
            pass

    def test_substract():

        array1 = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        array2 = [[1, 2], [3, 4], [5, 6]]
        array3 = [1,2,3]
        array4 = [1,2,3,4]

        assert snp.substract(array2, array2) == [[0, 0], [0, 0], [0, 0]]
        assert snp.substract(array3, array3) == [0,0,0]

        #Checks wether shape mismatch between matrices gives an error.
        try:
            snp.substract(array1, array2)
            raise Exception("Error")
        
        except Exception:
            pass
        
        #Checks wether substractions of matrices with vectors gives an error.
        try:
            snp.substract(array3, array1)
            raise Exception("Error")
        
        except Exception:
            pass
        
        #Checks wether shape mismatch within vectors gives an error.
        try:
            snp.substract(array4, array3)
            raise Exception("Error")
        
        except Exception:
            pass

    def test_dot_product():
        array1 = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        array2 = [[1, 2], [3, 4], [5, 6]]
        array3 = [[1, 2, 3], [4, 5, 6]]
        array4 = [1,2,3]
        array5 = [1,2,3,4]

        assert snp.dot_product(array2, array3) == [[9, 12, 15], [19, 26, 33], [29, 40, 51]]

        assert snp.dot_product(array4, array2) == [22,28]

        assert snp.dot_product(array4, array4) == [14]

        #Checks wether shape mismatch between matrices gives an error.
        try:
            snp.dot_product(array1, array2)
            raise Exception("Error")
        
        except:
            pass

        #Checks wether shape mismatch gives an error.
        try:
            snp.dot_product(array2, array4)
            raise Exception("Error")
        
        except:
            pass
        
        #Checks wether shape mismatch within vectors gives an error.
        try:
            snp.dot_product(array4, array5)
            raise Exception("Error")
        
        except:
            pass
    
    def test_gaussian_solver():
        array1 = [[1, 2], [3, 5]]
        array2 = [5, 13]
        array3 = [[1, 2, 3], [2, 4, 7]]


        assert snp.gaussian_solver(array1, array2) == [1,2]

        #Checking if singular matrices give error.
        try:
            snp.dot_product(array3, array2)
            raise Exception("Error")
        
        except:
            pass

        #Checking that if the first array is not a matrix it gives an error.
        try:
            snp.dot_product(array2, array2)
            raise Exception("Error")
        
        except:
            pass
        
        #Checking that if the second array is not a vector it gives an error.
        try:
            snp.dot_product(array1, array1)
            raise Exception("Error")
        
        except:
            pass

        #Checking shape mismatch.
        try:
            snp.dot_product(array3, array2)
            raise Exception("Error")
        
        except:
            pass

    def run_test():
        tests.test_is_valid_array()
        tests.test_ones()
        tests.test_zeros()
        tests.test_get()
        tests.test_append()
        tests.test_shape()
        tests.test_reshape()
        tests.test_add()
        tests.test_substract()
        tests.test_dot_product()
        tests.test_gaussian_solver()
        print("All tests passed!")

def main():
    tests.run_test()

if __name__ == "__main__":
    tests.run_test()

######################################################################################################
############################## Question 2 ############################################################
######################################################################################################


import numpy as np

class HammingCode(object):   

    def __init__(self, number):
        '''
        Initializes the HammingCode class
        name: str
        '''
        
        self.number=int(number) 
        self.binary_value= None
        self.message_vector = None
        self.encoded = None
        self.wrongcoded = None
        self.correctedcoded = None
        self.checkvector_wrongcoded = None


        self.convert_number_to_binary()

    def convert_number_to_binary(self):
        '''
        Every number from 0 to 15 can be transformed into a 4 digit binary number 
        '''
        if self.number >= 0 and self.number <= 15:
            self.binary_value= format(self.number, '04b')
            print(f'Integer Number: {self.number}. Binary Equivalent (4 digits): {self.binary_value}')
        else:
            raise Exception ('The number provided is not in the range 0-15 and therefore cannot be transformed into a 4-digit binary value') 
          
    def number_to_vector(self):
        '''
        Create a vector from the individual binary digits and check again it's four digits and binary
        '''
        self.message_vector = [int(digit) for digit in self.binary_value]
        if len(self.message_vector) != 4:
            raise Exception("The number provided does not have four digit")
        for element in self.message_vector:
            if element > 1 or element <0:
                raise Exception("The number provided is not a binary value")   
        self.message_vector = np.array([self.message_vector]) 
        print(f'The Message Vector for number {self.number} is {self.message_vector}')

    def encode(self):
        '''
        By multiplying Matrix G (7,4) for the message vector (1,4) we obtain the encoded vector of our message (1,7)
        '''
        G = np.array([[1, 1, 0, 1],
            [1, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        G = np.transpose(G)

        # Multiplying the vector (1,4) for the matrix transpose (4,7) so to be able to do the operation and reduced for modulo 2 so to have it binary
        self.encoded = np.dot(self.message_vector, G) % 2 
        print(f'The Encoded Vector for number {self.number} is {self.encoded}')

    def introduce_error(self, number_of_errors, error1_position, error2_position=False): 
        '''
        Possibility to add either one error digit or two error digits to the encoded message
        '''
        if error2_position:
            if error2_position < 0 or error2_position > 7:
                raise Exception("There position of error number 2 is not a valid position because it is not between 0 and 7 included")
            if error2_position == error1_position:
                raise Exception("The position of error number 2 cannot be the same as the position of error number 1")
        if error1_position < 0 or error1_position > 7:
            raise Exception("There position of error number 1 is not a valid position because it is not between 0 and 7 included")
        if self.encoded is None:
            raise Exception("There is no encoded message")
        elif number_of_errors != 1 and number_of_errors != 2:
            raise Exception("the number of errors is neither 1 nor 2")
        else:
            self.wrongcoded=self.encoded.copy()
            if number_of_errors == 1:
                #Introduce error by flipping the digit at the position of error number 1
                self.wrongcoded[0,error1_position-1]=1-self.wrongcoded[0,error1_position-1]
                print(f'Introduced error in one digit; the wrong vector is {self.wrongcoded}')
            else: 
                #Introduce errors by flipping the two digits at the positions of error number 1 and 2
                self.wrongcoded[0,error1_position-1]=1-self.wrongcoded[0,error1_position-1]
                self.wrongcoded[0,error2_position-1]=1-self.wrongcoded[0,error2_position-1]
                print(f'Introduced error in two digits; the wrong vector is {self.wrongcoded}')        
    
    '''
    Then the decoding part starts, considering how usually one end is doing the encoding and another end is doing the decoding,
    we consider how the party doing the decoding simply receives the encoded message and therefore has the need to check 
    if the number is seven digit and binary so to be decoded
    '''

    def checkencoded(self, check_wrongcoded = False):
        '''
        Checking if the message received is binary and seven digits, if so then it's possible to proceed with the decodification
        '''
        if check_wrongcoded:
            if self.wrongcoded.size != 7:
                raise Exception("The number provided does not have seven digit")
            for element in self.wrongcoded: 
                if (self.encoded > 1).any() or (self.wrongcoded < 0).any():
                    raise Exception("The number provided is not a binary value")
                print(f'The Encoded Message Vector {self.wrongcoded} is a binary seven digit value')
        else:
            if self.encoded.size != 7:
                raise Exception("The number provided does not have seven digit")
            for element in self.encoded: 
                if (self.encoded > 1).any() or (self.encoded < 0).any():
                    raise Exception("The number provided is not a binary value")
                print(f'The Encoded Message Vector {self.encoded} is a binary seven digit value')


    def paritycheck(self, check_wrongcoded = False):
        '''
        Checking in which position the encoded message is carrying error (if it's carrying any) by multiplying matrix H and the encoded vector, 
        and we assume the error is only one since the code can detect up to two bits errors but cannot identify the position for both 
        and correction is possible only when assuming only one bit is wrong
        '''
        H = np.array([[1,0,0],
              [0,1,0],
              [1,1,0],
              [0,0,1],
              [1,0,1],
              [0,1,1],
              [1,1,1]])
        
        #matrix H [7,3] and encoded [1,7]
        if check_wrongcoded:
            checkvector=np.dot(self.wrongcoded,H) % 2
            self.checkvector_wrongcoded = checkvector
        else: 
            checkvector=np.dot(self.encoded,H) % 2 #encoded [1,7] * H [7,3]

        if np.array_equal(checkvector, np.array([[0,0,0]])):
            print(f'The Check Vector is {checkvector} and no error is present')
        else: 
            print(f'The Check Vector is {checkvector} and we assume some error is present')

    
    def correct_error(self):
        """Assuming the error i a 1-bit error, it corrects the error and saves the corrected result
        """

        paritycheckdict = {0:np.array([[1,0,0]]), 1:np.array([[0,1,0]]), 2:np.array([[1,1,0]]), 3:np.array([[0,0,1]]), 4:np.array([[1,0,1]]), 5:np.array([[0,1,1]]), 6:np.array([[1,1,1]])}
        #Finding the correspondence between self.checkvector_wrongcoded and the arrays in paritycheckdict
        if self.wrongcoded is None:
            raise Exception("No error was created, therefore there is nothing to correct")
        a = None
        for key, value in paritycheckdict.items():   
            if np.array_equal(self.checkvector_wrongcoded,value):
                a = value.copy()
        #Finding the key corresponding to the value saved in the variable a
        if a is not None:
            for key, value in paritycheckdict.items():
                if np.array_equal(value,a):
                    error_index = key
                    print (f"the error is in position {error_index+1}")
                    break
            #Correcting the error
            self.correctedcoded = self.wrongcoded.copy()
            self.correctedcoded[0,error_index]=1-self.correctedcoded[0,error_index]
            print(f"If the error was 1-bit, the error in {self.wrongcoded} has been corrected. The restored encoded vector is {self.correctedcoded}. Otherwise a 2-digit error was converted in a 3-digit error, and the codeword retreived ({self.correctedcoded}) contains the wrong message")

        else: raise Exception("No one-bit error was found")

    def decode(self, restored=False):
        '''
        Using matrix R to decode the message to obtain the initial 4-bits vector and convert it back to the original integer number.
        This is done only for correct message to decode with no errors, because trying to decode something with errors would make no 
        sense and would not return a proper result.
        '''
        #Define R matrix
        R = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
        #Decoding
        if restored:
            codeword=np.array([self.correctedcoded])
        else:
            codeword=np.array([self.encoded])
        decoded=np.dot(codeword, R) % 2
        #Restores the decoded vector into a string and then an integer of 2 digits
        original_number = int(''.join(map(str, decoded.flatten())), 2) 
        print(f'The Decoded Message Vector is {decoded.flatten()}. The Original Integer Number inputed was {original_number}')
              

#Some examples of testing the different possible combinations


#Examples of encoding, parity checking and decoding with 0 error present
message_1= HammingCode(0)
message_1.number_to_vector()
message_1.encode()
message_1.checkencoded()
message_1.paritycheck()
message_1.decode()

message_2= HammingCode(2)
message_2.number_to_vector()
message_2.encode()
message_2.checkencoded()
message_2.paritycheck()
message_2.decode()

message_3= HammingCode(10)
message_3.number_to_vector()
message_3.encode()
message_3.checkencoded()
message_3.paritycheck()
message_3.decode()

message_4= HammingCode(15)
message_4.number_to_vector()
message_4.encode()
message_4.checkencoded()
message_4.paritycheck()
message_4.decode()

#Examples of encoding, parity checking and decoding with 1 digit error present
message_5= HammingCode(0)
message_5.number_to_vector()
message_5.encode()
message_5.introduce_error(1,3)
message_5.checkencoded(True)
message_5.paritycheck(True)
message_5.correct_error()
message_5.decode(True)

message_6= HammingCode(4)
message_6.number_to_vector()
message_6.encode()
message_6.introduce_error(1,1)
message_6.checkencoded(True)
message_6.paritycheck(True)
message_6.correct_error()
message_6.decode(True)

message_7= HammingCode(12)
message_7.number_to_vector()
message_7.encode()
message_7.introduce_error(1,7)
message_7.checkencoded(True)
message_7.paritycheck(True)
message_7.correct_error()
message_7.decode(True)

message_8= HammingCode(15)
message_8.number_to_vector()
message_8.encode()
message_8.introduce_error(1,6)
message_8.checkencoded(True)
message_8.paritycheck(True)
message_8.correct_error()
message_8.decode(True)

#Examples of encoding and checking with 2 digit error present. No correction will be attempted as by default it is impossible to correct a 2-digit error
message_9= HammingCode(0)
message_9.number_to_vector()
message_9.encode()
message_9.introduce_error(2,2,6)
message_9.checkencoded(True)
message_9.paritycheck(True)
message_9.correct_error()

message_10= HammingCode(6)
message_10.number_to_vector()
message_10.encode()
message_10.introduce_error(2,6,2)
message_10.checkencoded(True)
message_10.paritycheck(True)
message_10.correct_error()

message_11= HammingCode(11)
message_11.number_to_vector()
message_11.encode()
message_11.introduce_error(2,7,1)
message_11.checkencoded(True)
message_11.paritycheck(True)
message_11.correct_error()

message_12= HammingCode(14)
message_12.number_to_vector()
message_12.encode()
message_12.introduce_error(2,4,5)
message_12.checkencoded(True)
message_12.paritycheck(True)
message_12.correct_error()


######################################################################################################
############################## Question 3 ############################################################
######################################################################################################

import numpy as np
import os 
import re
import pandas as pd

class DocumentSimilarity: 

    def __init__(self):
        self.word_bank = set()
        self.words_in_each_document = {}
        self.binary_vectors = {}
        self.frequency_vectors = {}
    
    def read_doc(self, filepath: str) -> str:
        """
        Reads the document from the given file path and returns it as a string.
        """
        try:
            with open(filepath, 'r', encoding = 'utf-8') as file_read:
                doc = file_read.read()
            return doc
        except:
            raise Exception (f"There was an error reading the file {filepath}")
    
    def add_doc(self, filepath: str) -> str:
        """
        Creates the dictionary of valid words for each document as well as adding the new words to the word bank.
        """
        file_name = os.path.splitext(os.path.basename(filepath))[0]

        #Reads the document from the given file path and returns it as a string.
        doc_as_string = self.read_doc(filepath)

        #Extracts all of the words from the document string and cleans them.
        cleaned_words = self.get_cleaned_words(doc_as_string)

        #We append to the dictionary the name of the file as the key and the list of words as the value.
        self.words_in_each_document[file_name] = cleaned_words

        #We add the new words to a set that has every word that appears in any document.
        self.word_bank = self.word_bank.union(set(cleaned_words))

    def read_all_files(self, folder_path: str):
        """
        Reads every text document in the folder and executes add_doc function for each
        """
        #Checks if the folder exists and if it is accessible.
        try:
            os.listdir(folder_path)

        except Exception as e:
            raise Exception(f"There was an error reading the directory {folder_path}, error: {str(e)}")
        
        #For every file in the directory we execute add_doc function for it and check for errors.
        if len(os.listdir(folder_path)) != 0:
            for file in os.listdir(folder_path):
                if file.endswith(".txt"):
                    filepath = os.path.join(folder_path, file)
                    self.add_doc(filepath)
            
            txt_files = [file for file in os.listdir(folder_path) if file.endswith(".txt")]

            if len(txt_files) == 0:
                raise Exception("No .txt files found in the directory")
            if len(txt_files) == 1:
                raise Exception("The directory only contains one .txt file and therefore it cannot be compared to another")
        else:
            raise Exception ("The directory you provided is empty")
        
        self.update_word_vectors()

    def get_cleaned_words(self, doc_as_string: str) -> list:
        """
        Splits the text that has been read into different words and makes sure that there is no case sensitivity by converting all words to lowercase.
        """
        cleaned_words = re.sub('[^a-z]+', " ", doc_as_string.lower()).split()

        return cleaned_words
    
    def update_word_vectors(self):
        """
        Updates the two different word vectors for all documents:
        1. new_binary_vector: Vector of 1s and 0s that represent tif the word appears in the document.
        2. new_frequency_vector: Vector of the number of times the word appears in the document.
        """
        #For all documents in the corpus we create a new vector and a new vector frequency.
        for doc_name in self.words_in_each_document.keys():
            new_binary_vector = []
            new_frequency_vector = []

            for word in self.word_bank:
                if word in self.words_in_each_document[doc_name]:
                    new_binary_vector.append(1)
                    new_frequency_vector.append(self.words_in_each_document[doc_name].count(word))
                else: #If the word does not appear.
                    new_binary_vector.append(0)
                    new_frequency_vector.append(0)

            self.binary_vectors[doc_name] = new_binary_vector
            self.frequency_vectors[doc_name] = new_frequency_vector
    
    def dot_product_similarity(self, doc_name1: str, doc_name2: str) -> float:
        """
        calculates the dot product similarity between two vectors
        """
        vector_1 = np.array(self.binary_vectors[doc_name1])
        vector_2 = np.array(self.binary_vectors[doc_name2])
        return round(np.dot(vector_1,vector_2),2)
    
    def distance_norm_similarity(self, doc_name1: str, doc_name2: str) -> float:
        """
        calculates the euclidean norm similarity between two vectors
        """
        vector_1 = np.array(self.binary_vectors[doc_name1])
        vector_2 = np.array(self.binary_vectors[doc_name2])
        return round(np.linalg.norm(vector_1-vector_2),2)
    
    def get_cosine_similarity(self, doc_name1: str, doc_name2: str) -> float:
        """
        calculates the cosine similarity between two vectors
        """
        vector_1 = np.array(self.binary_vectors[doc_name1])
        vector_2 = np.array(self.binary_vectors[doc_name2])
        denominator = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
        return round(np.dot(vector_1,vector_2)/denominator,2)
    
    def get_cosine_similarity_frequency(self, doc_name1: str, doc_name2: str) -> float:
        """
        calculates the cosine similarity between two vectors, but taking into account the frequency in which a word appears in the text.
        """
        vector_1 = np.array(self.frequency_vectors[doc_name1])
        vector_2 = np.array(self.frequency_vectors[doc_name2])
        denominator = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
        return round(np.dot(vector_1,vector_2)/denominator,2)

    def get_similarity_all(self, chosen_doc_name: str, methods: list) -> float:
        """
        updates the dictionary containing the similarity scores between the documents with their similarity scores
        """
        dict_scores = {}

        #Updates the dictionary containing the similarity scores between the documents with their new similarity scores for all docs.
        for method in methods:
            dict_scores[method] = {}
            for doc_name in self.words_in_each_document.keys():
                if chosen_doc_name == doc_name:
                    continue
                if method == "cos":
                    dict_scores[method][doc_name] = self.get_cosine_similarity(chosen_doc_name, doc_name)
                if method == "freq-cos":
                    dict_scores[method][doc_name] = self.get_cosine_similarity_frequency(chosen_doc_name, doc_name)
                if method == "dot":
                    dict_scores[method][doc_name] = self.dot_product_similarity(chosen_doc_name, doc_name)
                if method == "norm":
                    dict_scores[method][doc_name] = self.distance_norm_similarity(chosen_doc_name, doc_name)
        
        dict_scores_df = pd.DataFrame(dict_scores)

        return dict_scores_df    
                                        
    def add_doc_compute_similarity(self, new_doc_path: str, methods: list):
        """
        We add the doc to the corpus (add_doc()), we update all the vectors, compute the similarity to all of them (with respect to the one added),
        and we rank the similarity and return a dict with the most similar docs.
        Methods are: cos, freq-cos, dot, norm.
        """
        #Checks if the method provided is valid.
        for method in methods:
            if method not in ["cos", "freq-cos", "dot", "norm"]:
                raise Exception ("Invalid method provided. The supported methods are: cos, freq-cos, dot, norm")
        
        file_name = os.path.splitext(os.path.basename(new_doc_path))[0]

        #If the file does not exist in the corpus we add it and update the vectors in this case.
        if file_name not in self.words_in_each_document.keys():
            self.add_doc(new_doc_path)
            self.update_word_vectors()

        return self.get_similarity_all(file_name,methods)


document_similarity = DocumentSimilarity()
folder_path = "Documents_Q3"

#Read and add all documents in the folder to the class.
document_similarity.read_all_files(folder_path)

#Adds DemocInn if not there already and computes the similarity score between DemocInn and the rest of the documents.
print(document_similarity.add_doc_compute_similarity('Documents_Q3/DemocInn.txt', ['freq-cos','cos','dot', 'norm']))

