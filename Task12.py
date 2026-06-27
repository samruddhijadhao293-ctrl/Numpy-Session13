print("Question 2:")

def analyze_string(s):

    if s == "":
        print("Empty string entered!")
        return

    # Length of string
    print("Length of string:", len(s))

    # Reverse using slicing
    print("Reverse string:", s[::-1])

    # Count vowels
    vowel_count = 0

    for ch in s.lower():
        if ch in "aeiou":
            vowel_count += 1

    print("Number of vowels:", vowel_count)

    # Print characters with positive and negative index
    print("\nCharacter with indexes:")

    for i in range(len(s)):
        print(
            "Character:", s[i],
            "Positive index:", i,
            "Negative index:", i-len(s)
        )

# User input
text = input("Enter a string: ")

# Function call
analyze_string(text)

print("Question 3:")
def manage_marks():

    marks = []

    # Taking 5 subject marks input from user
    for i in range(5):
        while True:
            try:
                mark = float(input(f"Enter marks for subject {i+1}: "))

                # Store marks in list
                marks.append(mark)
                break

            # Handle non-numeric input
            except ValueError:
                print("Invalid input! Please enter numeric marks.")

    # Calculate average, highest and lowest marks
    average = sum(marks) / len(marks)
    highest = max(marks)
    lowest = min(marks)

    print("\nMarks List:", marks)
    print("Average Marks:", average)
    print("Highest Marks:", highest)
    print("Lowest Marks:", lowest)

    # Sort list in descending order
    marks.sort(reverse=True)

    print("Marks in descending order:", marks)

# Function call
manage_marks()

print("Question  4:")

class Student:

    # Initialize student details
    def __init__(self, name, roll_no):
        self.name = name
        self.roll_no = roll_no
        self.marks_list = []

    # Add marks to list
    def add_mark(self, mark):
        try:
            mark = float(mark)

            if mark < 0 or mark > 100:
                raise ValueError

            self.marks_list.append(mark)
            print("Mark added successfully!")

        except ValueError:
            print("Invalid mark! Enter marks between 0 to 100.")

    # Calculate average marks
    def get_average(self):
        if len(self.marks_list) == 0:
            return 0

        return sum(self.marks_list) / len(self.marks_list)

    # Display student information
    def display_info(self):
        print("\nStudent Details")
        print("Name:", self.name)
        print("Roll No:", self.roll_no)
        print("Marks:", self.marks_list)
        print("Average:", self.get_average())


# Create student object using user input
try:
    name = input("Enter student name: ")
    roll = input("Enter roll number: ")

    student1 = Student(name, roll)

    # Taking marks input
    for i in range(5):
        mark = input(f"Enter mark {i+1}: ")
        student1.add_mark(mark)

    # Demonstrate methods
    student1.display_info()

except Exception as e:
    print("Something went wrong:", e)


print("Question 5:")
def student_database():
    students = {}

    while True:
        print("\n--- Student Database ---")
        print("1. Add Student")
        print("2. Search Student")
        print("3. Display All Students")
        print("4. Exit")

        try:
            choice = int(input("Enter choice: "))

            if choice == 1:
                roll = int(input("Enter roll number: "))

                name = input("Enter name: ")
                age = int(input("Enter age: "))
                city = input("Enter city: ")

                students.update({
                    roll: {
                        "name": name,
                        "age": age,
                        "city": city
                    }
                })

                print("Student added successfully!")

            elif choice == 2:
                roll = int(input("Enter roll number to search: "))

                student = students.get(roll)

                if student:
                    print("\nStudent Details:")
                    print("Name:", student.get("name"))
                    print("Age:", student.get("age"))
                    print("City:", student.get("city"))
                else:
                    print("Student not found!")

            elif choice == 3:
                if students:
                    print("\nAll Students:")

                    for roll, data in students.items():
                        print("\nRoll No:", roll)
                        print("Name:", data.get("name"))
                        print("Age:", data.get("age"))
                        print("City:", data.get("city"))
                else:
                    print("Database is empty!")

            elif choice == 4:
                print("Exiting...")
                break

            else:
                print("Invalid choice!")

        except ValueError:
            print("Please enter valid input!")

student_database()

print("Question 6:")
import random
import math

try:
    numbers = set()

    print("Enter 10 numbers:")

    for i in range(10):
        num = int(input(f"Enter number {i+1}: "))
        numbers.add(num)

    print("\nUnique numbers (Set):")
    print(numbers)

    # Convert set into tuple
    num_tuple = tuple(numbers)

    print("\nTuple:")
    print(num_tuple)

    # Pick 3 random numbers
    random_numbers = random.sample(num_tuple, 3)

    print("\n3 Random numbers:")
    print(random_numbers)

    # Square root of sum
    total = sum(num_tuple)

    print("\nSum of tuple elements:")
    print(total)

    print("Square root of sum:")
    print(math.sqrt(total))

except ValueError:
    print("Please enter only numbers!")

except Exception as e:
    print("Error occurred:", e)



print("Question 7:")
try:
    # Lambda function to calculate square
    square = lambda x: x * x

    # Generate numbers from 1 to 20
    numbers = range(1, 21)

    # Store squares in list using lambda
    squares = []

    for num in numbers:
        squares.append(square(num))

    print("All Squares:")
    print(squares)

    # Print only even squares
    even_squares = []

    for value in squares:
        if value % 2 == 0:
            even_squares.append(value)

    print("\nEven Squares:")
    print(even_squares)

except Exception as e:
    print("Error occurred:", e)



print("Question 8:")
class Employee:

    def __init__(self, emp_id, name, details):
        self.emp_id = emp_id
        self.name = name
        self.details = details   # tuple (department, salary)

    def show_details(self):
        print("\nEmployee ID:", self.emp_id)
        print("Name:", self.name)
        print("Department:", self.details[0])
        print("Salary:", self.details[1])

# Dictionary to store employee objects
employees = {}

# Adding 3 employees

emp1 = Employee(101, "Gunjan", ("IT", 50000))
emp2 = Employee(102, "Riya", ("HR", 45000))
emp3 = Employee(103, "Amit", ("Finance", 55000))

employees[101] = emp1
employees[102] = emp2
employees[103] = emp3

# Display all employees

print("Employee Details:")

for emp_id, emp_obj in employees.items():
    emp_obj.show_details()



print("Question 9:")
import math

try:
    # Take sentence input
    sentence = input("Enter a sentence: ")

    # Convert sentence into words
    words = sentence.split()

    # Store unique words using set
    unique_words = set(words)

    # Print sorted words
    sorted_words = sorted(unique_words)

    print("\nUnique words in sorted order:")
    for word in sorted_words:
        print(word)

    # Count unique words
    count = len(unique_words)

    print("\nTotal unique words:", count)

    # Power of 2 using math module
    result = math.pow(count, 2)

    print("Square of total unique words:", result)

except Exception as e:
    print("Error occurred:", e)



print("Question 10:")

import math
import random
from datetime import datetime

# Dictionary to store history
history = {}

# Basic arithmetic function
def arithmetic():
    try:
        a = float(input("Enter first number: "))
        b = float(input("Enter second number: "))

        print("1.Addition")
        print("2.Subtraction")
        print("3.Multiplication")
        print("4.Division")

        choice = int(input("Choose operation: "))

        if choice == 1:
            result = a + b

        elif choice == 2:
            result = a - b

        elif choice == 3:
            result = a * b

        elif choice == 4:
            result = a / b

        else:
            print("Invalid choice")
            return

        print("Result:", result)
        return result

    except ZeroDivisionError:
        print("Cannot divide by zero!")

    except ValueError:
        print("Enter valid numbers!")

# Scientific calculation
def scientific():
    try:
        num = float(input("Enter number: "))

        print("1. Square root")
        print("2. Power")

        choice = int(input("Choose: "))

        if choice == 1:
            result = math.sqrt(num)

        elif choice == 2:
            power = int(input("Enter power: "))
            result = math.pow(num, power)

        else:
            print("Invalid choice")
            return

        print("Result:", result)
        return result

    except Exception as e:
        print("Error:", e)

# Random number generator
def random_number():
    try:
        start = int(input("Start number: "))
        end = int(input("End number: "))

        result = random.randint(start, end)

        print("Random number:", result)
        return result

    except ValueError:
        print("Enter valid numbers!")

# Store result
def save_result(result):

    if result is not None:

        time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        history[time] = result

        print("Result saved!")

# View history
def view_history():

    if history:

        print("\n--- History ---")

        for time, result in history.items():
            print(time, ":", result)

    else:
        print("No history found")

# Main menu

while True:

    print("\n===== Smart Calculator & Data Manager =====")
    print("1. Basic Arithmetic")
    print("2. Scientific Calculations")
    print("3. Generate Random Numbers")
    print("4. Store Result")
    print("5. View History")
    print("6. Exit")

    try:
        choice = int(input("Enter choice: "))

        if choice == 1:
            result = arithmetic()

        elif choice == 2:
            result = scientific()

        elif choice == 3:
            result = random_number()

        elif choice == 4:
            save_result(result)

        elif choice == 5:
            view_history()

        elif choice == 6:
            print("Thank you!")
            break

        else:
            print("Invalid option")

    except ValueError:
        print("Please enter number only!")



