class HashTable(object):
    def __init__(self, size, hashfun, thresh=.75):
        # Allocating a list of nones
        self.hashtable = [None]*size
        self.size = size
        self.len = 0
        self.thresh = thresh
        # Hash function (define your function outside of the class, anything that defines a call method is callable)
        self._hash = hashfun
        # Ascertains that the load attribute is defined.
        self._update_load()

    def __len__(self):
        return self.len

    def _update_load(self):
        self.load = float(self.len)/self.size

    def __iter__(self):
        for b in self.hashtable:
            if b is not None:
                for i in b:
                    yield i

    def load(self):
        self._update_load()
        return self.load

    def _insert(self, data, hashtable, hashsize):
        hx = self._hash(data[0], hashsize)
        try:
            found = False
            for i, v in enumerate(hashtable[hx]):
                if v[0] == data[0]:
                    found = True
                    break
            if found:
                hashtable[hx][i] = data
            else:
                hashtable[hx].append(data)
        except (AttributeError, TypeError):
            hashtable[hx] = [data]

    def insert(self, data):
        self._insert(data, self.hashtable, self.size)
        self.len += 1
        self._update_load()

        if self.load > self.thresh:
            self._realloc(self.size*2)

    def _realloc(self, newsize):
        newhash = [None]*newsize

        for i in iter(self):
            self._insert(i, newhash, newsize)

        self.hashtable = newhash
        self.size = newsize
        self._update_load()

    def find(self, key):
        hx =  self._hash(key, self.size)
        try:
            found = False
            for i, v in enumerate(self.hashtable[hx]):
                if v[0] == key:
                    return v[1]
        except ValueError:
            raise

    def remove(self, data):
        hx = self._hash(data[0])
        try:
            self.hashtable[hx].remove(data)
        except ValueError:
            raise

