import itertools

class Session(object):
    def __init__(self):
        self.channels = {}
        self.register_channel(0)
        self.sessionlog = []
        
    def register_channel(self, n, topic=None):
        #check if channel already exists
        if n in self.channels:
            raise KeyError("Channel already exists!")
        
        self.channels[n] = topic
    
    def unregister_channel(self, n):
        try:
            del self.channels[n]
        except KeyError:
            pass
        
    def change_nick(self, ip, nick):
        self.clients[ip] = nick
        
    def get_channels(self):
        return self.channels
    
    def log_message(self, message):
        self.sessionlog.append(message)
        self.__emergency_purge__()
        
   
    
    def __emergency_purge__(self):
        if len(self.sessionlog) > 10000000:
            self.sessionlog = []