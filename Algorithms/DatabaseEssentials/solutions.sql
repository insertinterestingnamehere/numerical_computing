
--book
CREATE TABLE student_classes (StudentID INT NOT NULL, ClassID INT, Grade VARCHAR(2));
CREATE TABLE student_information (StudentID INT NOT NULL, Name VARCHAR(20), SocSecurity INT, MajorID INT);


--Problem One
CREATE TABLE major_information (MajorID INT, MajorName VARCHAR(20));
CREATE TABLE class_information (ClassID INT, ClassName VARCHAR(20));


--Problem two
.import students.dat student_information
.import classes.dat student_classes
.import class_info.dat class_information
.import major_info.dat major_information


--Problem 3
SELECT Name, MajorName,ClassName FROM student_information INNER JOIN student_classes ON student_information.StudentId = student_classes.StudentID INNER JOIN class_information ON student_classes.ClassID = class_information.ClassID INNER JOIN major_information ON student_information.MajorID = major_information.MajorID;

