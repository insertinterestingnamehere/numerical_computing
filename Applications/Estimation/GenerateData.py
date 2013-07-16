from random import randint

survey = []
for j in range(0,100):
    rank = []
    person = []
    r = randint(1,5)
    person.append(r)
    for i in range(0,28):
        stats = []
        #Rank
        r = randint(1,28)
        while r in rank:
           r = randint(1,28)
        stats.append(r)
        rank.append(r)
        #How much influence vote
        r = randint(1,5)
        stats.append(r)
        
        #For/Against
        r = randint(1,5)
        stats.append(r)
        
        person.append(stats)
    
    survey.append(person)

for row in survey:
    print row
