class PriorityQueue(object):
    def __init__(self):
        self.queue = []
        self.size = 0

    def push(self, value, priority):
        self.queue.append((priority, value))
        self.queue.sort()
        self.size += 1

    def pop(self):
        try:
            x = self.queue.pop(0)[1]
            self.size -= 1
            return x
        except (IndexError, e):
            raise IndexError(e)

    def peek(self):
        return self.queue[0]
