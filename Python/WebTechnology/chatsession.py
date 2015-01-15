import itertools
from bisect import bisect_left, bisect_right

class Session(object):
    def __init__(self):
        self.channels = {}
        self.register_channel(0)
        self.sessionlog = []
        self.users = set()
        
    def register_channel(self, n, topic=None):
        #check if channel already exists
        if n in self.channels:
            raise KeyError("Channel already exists!")
        
        self.channels[n] = topic
    
    def get_channels(self):
        return self.channels
    
    def log_message(self, message):
        if message['content'].strip():
            self.sessionlog.append(message)
            self.sessionlog.sort(key=lambda x: x['timestamp'])
            
            #update user's timestamp
            #self.users[message['nick']] = message['timestamp']
        self.__emergency_purge__()
        
    def retrieve_new(self, nick, timestamp, channel):
        def binary_search(t):
            '''return index of first element that is >= t'''
            
            imin, imax = 0, len(self.sessionlog) - 1
            while imin < imax:
                imid = (imax+imin)//2
                print imin, imid, imax
                if self.sessionlog[imid]['timestamp'] < t:
                    imin = imid + 1
                else:
                    imax = imid
            if imin == imax and self.sessionlog[imin]['timestamp'] == t:
                return imin
            
        def filter_channel(message):
            if message['channel'] == channel:
                return True
            return False
        
        i = binary_search(timestamp)
        mfilter = itertools.ifilter(filter_channel, 
                                    itertools.islice(self.sessionlog, i+1))
        return list(mfilter)
    
    def update_user(self, user, timestamp):
        self.users[user] = timestamp
        
    def change_nick(self, old_nick, new_nick):
        if new_nick in self.users:
            raise KeyError
        if old_nick is not None:
            self.users.remove(old_nick)
        self.users.add(new_nick)
        
    def __emergency_purge__(self):
        if len(self.sessionlog) > 10000000:
            self.sessionlog = []