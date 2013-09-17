# Solutions provided by A. Zaitzeff BYU

#Builds the NetworkX graph
import networkx as nx
def buildBaconX(n):
    f = open('movieData.txt', 'r')
    G=nx.Graph()
    for i in xrange(n):
        info=f.readline().strip().split('/')
        size=len(info)
        for i in xrange(1,size):
            for j in xrange(1,size):
                if i!=j:
                    G.add_edge(info[i],info[j])
    f.close()
    return G

#The next three functions are used to find out what movie actors have in common. 
#The first builds a dictionary linking actors to the movies they have been in
def buildMovie(n):
    f = open('movieData.txt', 'r')
    actorMovie=dict()
    for i in xrange(n):
        info=f.readline().strip().split('/')
        actorMovie=addThem(info,actorMovie)
    f.close()
    return actorMovie

#helps the previous function
def addThem(info,actorMovie):
    size=len(info)
    for i in xrange(1,size):
        if not(actorMovie.has_key(info[i])):
            actorMovie[info[i]]=set()
        actorMovie[info[i]].add(info[0])
    return actorMovie

#given the output of buildMovie and a path of actors outputs the movies that connect the actors
def links(actorMovie,connect):
    link=[]
    size=len(connect)
    for i in xrange(size-1):
        link.append(connect[i])
        tempSet=actorMovie[connect[i]].intersection(actorMovie[connect[i+1]])
        link.append(tempSet.pop())
    link.append(connect[-1])
    return link

#solves Problem 1
n=1299
G= buildBaconX(n)
actorMovie=buildMovie(n)

#solves Problem 2 part 1
#Solution is in form [Actor,Movie,Actor,...,Movie,Actor]
ans1=nx.shortest_path(G,'Bacon, Kevin','Neeson, Liam')
path=links(actorMovie,ans1)
#All possible solutions are: (movies can be different)
#['Bacon, Kevin', 'R.I.P.D.', 'Strassmann, Franz', 'The Dark Knight Rises', 'Neeson, Liam']
#['Bacon, Kevin', 'Crazy, Stupid, Love.', 'King, Joey', 'The Dark Knight Rises', 'Neeson, Liam']
#['Bacon, Kevin', 'X-Men: First Class', 'Jones, January', 'Unknown', 'Neeson, Liam']
#['Bacon, Kevin', 'Crazy, Stupid, Love.', 'Lee, Reggie', 'The Dark Knight Rises', 'Neeson, Liam']
#['Bacon, Kevin', 'X-Men: First Class', 'Atkins, Lasco', 'Wrath of the Titans', 'Neeson, Liam']
#['Bacon, Kevin', 'R.I.P.D.', 'Perry, Steve', 'The Dark Knight Rises', 'Neeson, Liam']
#['Bacon, Kevin', 'R.I.P.D.', 'Alejandro, Charlie', 'The Dark Knight Rises', 'Neeson, Liam']
#['Bacon, Kevin', 'X-Men: First Class', 'Serbedzija, Rade', 'Taken 2', 'Neeson, Liam']
#['Bacon, Kevin', "Jayne Mansfield's Car", 'Briscoe, Brent', 'The Dark Knight Rises', 'Neeson, Liam']


#solves Problem 2 part 2
ans1=nx.shortest_path(G,'Bacon, Kevin','Zahid, Imran')
path=links(actorMovie,ans1)
#one possible solutions is: ['Bacon, Kevin','X-Men: First Class','Grant, Corin','Magic Mike','Banfield, Cameron','Beneath the Darkness',
#'Sam','Dum Maaro Dum','Basu, Bipasha', 'Raaz 3: The Third Dimension','Hashmi, Emraan','Jannat 2','Zahid, Imran']


#solves Problem 3
allLen=nx.shortest_path_length(G,'Bacon, Kevin')
values=allLen.values()
avg=sum(values)/(1.*len(values));avg


numberWithoutKevinBaconNumber=len(actorMovie.keys())-len(values)
howmany={0:0,1:0,2:0,3:0,4:0,5:0,6:0,'inf':numberWithoutKevinBaconNumber}
for i in values:
    howmany[i]+=1
#Solution
#The average Bacon number is 2.66
#How many people have each Bacon number {0: 1, 1: 345, 2: 15819, 3: 24235, 4: 1977, 5: 130, 6: 6, 'inf': 847} 