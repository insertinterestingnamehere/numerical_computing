from collections import deque

def bfs(G, v, d, track=True):
    Q = deque([v])
    marked = set([v])
    visited = list()
    while len(Q) > 0:
        t = Q.popleft()
        visited.append(t)
        if t == d:
            return t, visited
        else:
            for e in adjedges_list(G, t):
                if track:
                    if e not in marked:
                        marked.add(e)
                        Q.append(e)
                else:
                    Q.append(e)
    
def dfs(G, v, d, track=True):
    Q = list([v])
    marked = set([v])
    visited = list()    
    while len(Q) > 0:
        t = Q.pop()
        visited.append(t)
        if t == d:
            return t, visited
        else:
            for e in adjedges_list(G, t):
                if track:
                    if e not in marked:
                        marked.add(e)
                        Q.append(e)
                else:
                    Q.append(e)
                        
def adjedges_mat(M, e):
    for i, n in enumerate(M[e]):
        if n == 1:
            yield i

def adjedges_list(M, e):
    for n in M[e]:
        yield n