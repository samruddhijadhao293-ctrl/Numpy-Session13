import numpy as np
print("Question 1: ")
# Create a 2D array of shape (5, 6) with random integers between 10 and 100
arr = np.random.randint(10, 101, (5, 6))

# Print the array
print("Array:")
print(arr)

# Print properties
print("\nShape of the array:", arr.shape)
print("Total number of elements (size):", arr.size)
print("Data type (dtype):", arr.dtype)


import numpy as np
print("Question 2: ")
# Generate a 1D array of 20 random numbers between 1 and 50
arr = np.random.randint(1, 51, 20)

# Print the array
print("Array:")
print(arr)

# Minimum value and its index
print("\nMinimum value:", arr.min())
print("Index of minimum value:", arr.argmin())

# Maximum value and its index
print("\nMaximum value:", arr.max())
print("Index of maximum value:", arr.argmax())

# Sum of all elements
print("\nSum of all elements:", arr.sum())

# Mean and Standard Deviation
print("Mean:", arr.mean())
print("Standard Deviation:", arr.std())


import numpy as np
print("Question 3: ")
# Create a 4x5 matrix with random integers between 20 and 80
arr = np.random.randint(20, 81, (4, 5))

print("Matrix:")
print(arr)

# Statistics
print("\nMinimum value:", arr.min())
print("Maximum value:", arr.max())
print("Sum of all elements:", arr.sum())
print("Mean:", arr.mean())
print("Standard Deviation:", arr.std())

# Row-wise sum
print("\nSum of each row:")
print(arr.sum(axis=1))

# Column-wise sum
print("\nSum of each column:")
print(arr.sum(axis=0))


import numpy as np
print("Question 4: ")
# Create a 1D array
arr = np.arange(1, 25)

print("Original Array:")
print(arr)

# Reshape to (4,6)
a = arr.reshape(4, 6)
print("\n4 x 6 Array:")
print(a)
print("Shape:", a.shape)

# Reshape to (3,8)
b = arr.reshape(3, 8)
print("\n3 x 8 Array:")
print(b)
print("Shape:", b.shape)

# Reshape to (2,3,4)
c = arr.reshape(2, 3, 4)
print("\n2 x 3 x 4 Array:")
print(c)
print("Shape:", c.shape)


import numpy as np
print("Question 5: ")
arr = np.array([[10,20,30,40],
                [50,60,70,80],
                [90,100,110,120]])

print("Original Array:")
print(arr)

# First row
print("\nFirst Row:")
print(arr[0])

# Last column
print("\nLast Column:")
print(arr[:,3])

# Center 2x2 submatrix
print("\nCenter 2x2 Submatrix:")
print(arr[1:3,1:3])

# Even numbers
print("\nEven Numbers:")
print(arr[arr % 2 == 0])

import numpy as np
print("Question 6: ")
arr = np.random.randint(1,101,(5,5))

print("Original Matrix:")
print(arr)

# Diagonal elements
print("\nDiagonal Elements:")
print(np.diag(arr))

# Elements greater than 50
print("\nElements Greater Than 50:")
print(arr[arr > 50])

# Replace values less than 30 with 0
arr[arr < 30] = 0

print("\nModified Matrix:")
print(arr)



import numpy as np
print("Question 7: ")
a = np.array([10,20,30,40])
b = np.array([1,2,3,4])

print("Addition:")
print(a+b)

print("\nSubtraction:")
print(a-b)

print("\nMultiplication:")
print(a*b)

print("\nDivision:")
print(a/b)

print("\nPower:")
print(a**b)

print("\nModulo:")
print(a%b)



import numpy as np
print("Question 8: ")
A = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])

B = np.array([[9,8,7],
              [6,5,4],
              [3,2,1]])

# Element-wise multiplication
print("Element-wise Multiplication:")
print(A * B)

# Matrix multiplication
print("\nMatrix Multiplication:")
print(A @ B)

# Difference:
# * performs multiplication of corresponding elements.
# @ performs actual matrix multiplication.



import numpy as np
print("Question 9: ")
arr = np.random.randn(6,6)

print("Matrix:")
print(arr)

print("\nShape:", arr.shape)
print("Size:", arr.size)
print("Data Type:", arr.dtype)

print("\nIndex of Maximum Value:")
print(np.unravel_index(arr.argmax(), arr.shape))

print("\nIndex of Minimum Value:")
print(np.unravel_index(arr.argmin(), arr.shape))

print("\nTop Left 3x3 Matrix:")
print(arr[:3,:3])

# Replace negative numbers with absolute values
arr[arr < 0] = np.abs(arr[arr < 0])

print("\nModified Matrix:")
print(arr)

print("\nMean of Modified Matrix:")
print(arr.mean())



import numpy as np
print("Question 10: ")
# Generate marks of 10 students and 5 subjects
marks = np.random.randint(30,101,(10,5))

print("Student Marks:")
print(marks)

# Calculate total marks
total = marks.sum(axis=1)

# Calculate average marks
average = marks.mean(axis=1)

print("\nTotal Marks:")
print(total)

print("\nAverage Marks:")
print(average)

# Highest scorer
highest = total.argmax()

# Lowest scorer
lowest = total.argmin()

print("\nHighest Scoring Student Index:", highest)
print("Marks:", marks[highest])

print("\nLowest Scoring Student Index:", lowest)
print("Marks:", marks[lowest])

# Class statistics
print("\nOverall Class Mean:", marks.mean())
print("Overall Class Standard Deviation:", marks.std())

# Demonstrate reshape
reshaped = marks.reshape(5,10)

print("\nReshaped Array (5 x 10):")
print(reshaped)

# Top 3 students
top3 = total.argsort()[-3:][::-1]

print("\nTop 3 Students:")
print(marks[top3])

# Comments:
# sum(axis=1) -> Calculates total marks for each student.
# mean(axis=1) -> Calculates average marks.
# argmax() -> Finds highest scorer.
# argmin() -> Finds lowest scorer.
# argsort() -> Sorts totals and helps find top students.
# reshape() -> Changes array shape without changing data.

