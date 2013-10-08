# This is a modified version of the solution
# to the first problem in the MST lab.
def kruskal(edges, nodelist, div):
    removed = []
    nodes = {node:node for node in nodelist}
    remaining = len(nodes) - 1
    end = len(nodes) - div
    def track(node):
        temp = node
        while nodes[temp] is not temp:
            temp = nodes[temp]
        return temp
    for n1, n2, weight in sorted(edges, key=itemgetter(2)):
        root = track(n1)
        remove = track(n2)
        if root is not remove:
            removed.append(remove)
            remaining -=1
            if remaining == 0:
                return {node:track(node) for node in nodes}
            nodes[remove] = root