print("Question No. 1")
import numpy as np
# 1D Array
arr1 = np.array([10, 20, 30, 40, 50])

# 2D Array
arr2 = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

print("1D Array:")
print(arr1)
print("Shape:", arr1.shape)
print("Data Type:", arr1.dtype)

print("\n2D Array:")
print(arr2)
print("Shape:", arr2.shape)
print("Data Type:", arr2.dtype)

print("Question No. 2")
import numpy as np
# 1D array of zeros
a = np.zeros(8)

# 2D array of ones
b = np.ones((4,4))

# 3x3 matrix of zeros
c = np.zeros((3,3))

print("1D Array of Zeros:")
print(a)

print("\n2D Array of Ones:")
print(b)

print("\n3x3 Matrix of Zeros:")
print(c)

print("Question No. 3")
import numpy as np
a = np.arange(0,21,1)
b = np.arange(10,51,2)
c = np.arange(5,101,5)

print("Numbers from 0 to 20:")
print(a)

print("\nEven Numbers from 10 to 50:")
print(b)

print("\nNumbers from 5 to 100 (Step 5):")
print(c)

print("Question No. 4")
import numpy as np
a = np.linspace(0,5,10)
b = np.linspace(-10,10,15)

print("10 Equally Spaced Numbers:")
print(a)
print("Length:", len(a))

print("\n15 Equally Spaced Numbers:")
print(b)
print("Length:", len(b))

print("Question No. 5")
import numpy as np
# Random numbers between 0 and 1
a = np.random.rand(10)

# Standard normal distribution
b = np.random.randn(3,3)

# Random integers
c = np.random.randint(10,51,size=(4,5))

print("1D Random Array:")
print(a)

print("\n3x3 Standard Normal Matrix:")
print(b)

print("\n4x5 Random Integer Matrix:")
print(c)

print("Question No. 6")
import numpy as np
v1 = np.array([2, 4, 6, 8])
v2 = np.array([1, 3, 5, 7])

print("Vector 1:", v1)
print("Vector 2:", v2)

print("\nAddition:")
print(v1 + v2)

print("\nSubtraction:")
print(v1 - v2)

print("\nElement-wise Multiplication:")
print(v1 * v2)

print("\nDot Product:")
print(np.dot(v1, v2))




print("Question No. 7")
import numpy as np
A = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])

B = np.array([[9,8,7],
              [6,5,4],
              [3,2,1]])

print("Matrix A:")
print(A)

print("\nMatrix B:")
print(B)

print("\nMatrix Addition:")
print(A + B)

print("\nMatrix Multiplication:")
print(np.dot(A, B))

print("\nElement-wise Multiplication:")
print(A * B)

print("Question No. 8")
import numpy as np
arr = np.random.randint(1,101,size=(4,4))

print("Array:")
print(arr)

print("\nShape:", arr.shape)
print("Dimension:", arr.ndim)
print("Total Elements:", arr.size)
print("Data Type:", arr.dtype)
print("Minimum Value:", arr.min())
print("Maximum Value:", arr.max())

print("Question No. 9")
import numpy as np
arr = np.random.randint(1,51,20)

print("1D Array:")
print(arr)

matrix = arr.reshape(4,5)

print("\n4x5 Matrix:")
print(matrix)

print("\nSum:", np.sum(matrix))
print("Mean:", np.mean(matrix))
print("Standard Deviation:", np.std(matrix))

print("\nMaximum Value in Each Row:")
print(np.max(matrix, axis=1))





print("Question No. 10")
import numpy as np
try:
    n = int(input("How many numbers do you want to generate? "))

    arr = np.random.randint(10,101,n)

    print("\nGenerated Array:")
    print(arr)

    print("\nMean:", np.mean(arr))
    print("Median:", np.median(arr))
    print("Standard Deviation:", np.std(arr))
    print("Minimum:", np.min(arr))
    print("Maximum:", np.max(arr))

    if n % 2 == 0:
        matrix = arr.reshape(2, n//2)
        print("\n2D Array:")
        print(matrix)

        print("\nRow-wise Sum:")
        print(np.sum(matrix, axis=1))
    else:
        print("\nArray cannot be reshaped into a 2D matrix because the number of elements is odd.")

except ValueError:
    print("Invalid Input! Please enter an integer.")
except Exception as e:
    print("Error:", e)
