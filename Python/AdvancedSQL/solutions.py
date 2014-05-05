import sqlite3 as sql
import random
import csv

dbfile = 'students.db'    
def get_conn():
    con = sql.connect(dbfile)
    return con

# prob:tablerelations (= one to one, =>/<= one to many/many to one, <=> many to many)
# StudentID=Name
# Name=>Grade (using grades)
# Name=>Classes (using classes)
# Name<=MajorCode (using majors)
# Name<=MinorCode (using majors)
# Majors.ID = Majors.Name
# Classes.ID = Classes.Name
# Grades.ClassID<=>Grades.Grade (Using Grades and Classes)
    
def studentmajors():
    con = get_conn()
    cur = con.cursor()
    
    try:
        cur.execute("select majors.name, count(students.name) from students left outer join majors on students.majorcode=majors.id group by students.majorcode order by majors.name asc;")
        results = cur.fetchall()
        cur.close()
        return results
    except:
        pass
    finally:
        con.close()
        
def studentGPA():
    con = get_conn()
    cur = con.cursor()
    
    try:
        cur.execute("""select round(sum(
                        case when grade in ('A+', 'A', 'A-') then 4.0
                            when grade in ('B+', 'B', 'B-') then 3.0
                            when grade in ('C+', 'C', 'C-') then 2.0
                            when grade in ('D+', 'D', 'D-') then 1.0
                            else 0.0
                        end)/count(*), 2) as grade
                    from students join grades on students.studentid=grades.studentid 
                    where grade is not NULL;""")
        result = cur.fetchall()[0]
        cur.close()
        return results
    except:
        pass
    finally:
        con.close()
        
def likec():
    con = get_con()
    cur = con.cursor()
    
    try:
        cur.execute("""select name, majorcode from students where name like '% C%';""")
        results = cur.fetchall()
        cur.close()
        return results
    except:
        pass
    finally:
        con.close()
