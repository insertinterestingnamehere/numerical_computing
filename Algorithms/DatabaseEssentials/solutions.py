import sqlite3 as sql
from collections import Counter

dbfile = 'icd9.db'
def get_conn():
    con = sql.connect(dbfile)
    return con

def new_tables():
    con = get_conn()
    try:
        cur = con.cursor()
        cur.execute('create table MajorInfo (MajorCode int, MajorName text);')
        cur.execute('create table CourseInfo (CourseID int, CourseName text);')
        cur.close()
        con.commit()
    except:
        con.rollback()
    finally:
        con.close()
        
def icd9tables():
    def read_dats():
        with open('icd9_data.csv', 'r') as F:
            dats = csv.csvreader(F, delimiter=',')
            for row in dats:
                yield row
    
    con = get_conn()
    try:
        cur = con.cursor()
        cur.execute('drop table icd9_problem;')
        cur.execute('create table icd9_problem(id int not null, gender text, age int, codes text);')
        idat = read_dats()
        cur.executemany('insert into icd9_problem(id, gender, age, codes), values (?, ?, ?, ?);', idat)
        cur.close()
        con.commit()
    except:
        con.rollback()
    finally:
        con.close()
    
def sampletables():
    creates = ["create table students (StudentID int not null, Name text, MajorCode text);",
               "create table majors (ID int not null, Name text);",
               "create table grades (StudentID int not null, ClassID int, Grade text);",
               "create table classes(ClassID int not null, Name text);"]
    
    data = {'students':((401767594, 'Michelle Fernandez', 1),
                        (678665086, 'Gilbert Chapman', 1),
                        (553725811, 'Roberta Cook', 2),
                        (886308195, 'Rene Cross', 3),
                        (103066521, 'Cameron Kim', 4),
                        (821568627, 'Mercedes Hall', 3),
                        (206208438, 'Kristopher Tran', 2),
                        (341324754, 'Cassandra Holland', 1),
                        (262019426, 'Alfonso Phelps', 3),
                        (622665098, 'Sammy Burke', 2)),
            'majors': ((1, 'Math'),
                       (2, 'Science'),
                       (3, 'Writing'),
                       (4, 'Art')),
            'grades': ((401767594, 4, 'C'),
                       (401767594, 3, 'B-'),
                       (678665086, 4, 'A+'),
                       (678665086, 3, 'A+'),
                       (553725811, 2, 'C'),
                       (678665086, 1, 'B'),
                       (886308195, 1, 'A'),
                       (103066521, 2, 'C'),
                       (103066521, 3, 'C-'),
                       (821568627, 4, 'D'),
                       (821568627, 2, 'A+'),
                       (821568627, 1, 'B'),
                       (206208438, 2, 'A'),
                       (206208438, 1, 'C+'),
                       (341324754, 2, 'D-'),
                       (341324754, 1, 'A-'),
                       (103066521, 4, 'A'),
                       (262019426, 2, 'B'),
                       (262019426, 3, 'C'),
                       (6226650980, 1, 'A'),
                       (622665098, 2, 'A-')),
            'classes': ((1, 'Calculus'),
                        (2, 'English'),
                        (3, 'Pottery'), 
                        (4, 'History'))}
            
    con = get_conn()
    try:
        cur = con.cursor()
        for x in creates:
            cur.execute(x)
        cur.executemany('insert into stduents values(?, ?, ?);', data['students'])
        cur.executemany('insert into majors values(?, ?);', data['majors'])
        cur.executemany('insert into grades values(?, ?, ?);', data['grades'])
        cur.executemany('insert into classes values(?, ?);', data['classes'])
        cur.close()
        con.commit()
    except:
        con.rollback()
    finally:
        con.close()

def youngfreqcodes():
    con = get_conn()
    cur = con.cursor()
    query = "select codes from icd9_problem where gender=? and age < 35 and age >= 25;"
    cur.execute(query, "M")
    
    MenCounter = Counter()
    mc = 0
    for code in cur:
        MenCounter.update(code[0].split(';'))
        mc += 1
    
    cur.execute(query, "F")
    WomenCounter = Counter()
    wc = 0
    for code in cur:
        WomenCounter.update(code[0].split(';'))
        wc += 1
        
    wmost = WomenCounter.most_common(1)
    mmost = MenCounter.most_common(1)
    
    cur.close()
    con.close()
    return mc, wc, mmost, wmost
