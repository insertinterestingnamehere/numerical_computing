import random
import csv
import collections as col
import itertools

Tribute = col.namedtuple('Tribute', 'name,district,gender')

def makepairs(males, females):
    random.shuffle(males)
    random.shuffle(females)

    tributes = []
    for d, t in enumerate(itertools.izip(males, females), 1):
        tributes.append(Tribute(t[0], d, "M"))
        tributes.append(Tribute(t[1], d, "F"))
        
    return tributes
    
def hasSurvived():
    if random.gauss(.5, .5) > .6:
        return True
    return False

def initGames(outFile):
    #read the tributes
    males = []
    females = []
    with open('tributes.csv', 'r') as f:
        for m, f in csv.reader(f):
            males.append(m.strip())
            females.append(f.strip())
    
    #read the events
    with open('events.txt', 'r') as f:
        events = [L for L in f]
        
    survivors = makepairs(males, females)
    simulateGames(survivors, events, outFile)
    
def simulateGames(survivors, events, outFile):
    days = itertools.count(1)
    d = days.next()
    with open(outFile, 'w') as out:
        while len(survivors) > 1:
            #choose the events
            out.write("Day {}\n".format(days))
            badthings = random.sample(events, len(survivors))
            
            survived = []
            for event, trib in itertools.izip(badthings, survivors):
                if hasSurvived():
                    survived.append(1)
                else:
                    survived.append(0)
                
                out.write("{} experienced {} and {}.\n".format(trib.name, event, 'survived' if survived[-1] == 1 else 'died'))
                            
            survivors = list(itertools.compress(survivors, survived))
            out.write("End of Day {}\n".format(days))
            out.write("Canon fired {} times\n\n".format(len(badthings) - sum(survived)))
            day = days.next()
            
            try:
                #remove False Start at end of Day 1
                events.remove("False Start")
            except ValueError:
                pass
        
        out.write("Day {}\n".format(days))
        if len(survivors) == 1:
            t = survivors[0]
            msg = "The final tribute was the {} from District {}: {}\n".format('girl' if t.gender == 'F' else 'boy', t.district, t.name)
        else:
            msg = "There are no surviving tributes.\n"
        
        out.write(msg)
        print msg,
if __name__ == "__main__":
    import sys
    initGames(sys.argv[1])