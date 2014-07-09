import json
from datetime import datetime
import requests
from datetime import datetime
from dateutil import parser
from uuid import uuid1

class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)

def dateobj(dct):
    if 'timestamp' in dct:
        parts = dct['timestamp'].split(".")
        tmp = datetime.strptime(parts[0], "%Y-%m-%dT%H:%M:%S")
        tmp.replace(microsecond=int(parts[1]))
        dct['timestamp'] = tmp
    return dct

class Session(object):
    def __init__(self, ip, port, nick):
        self.ip = ip
        self.port = port
        self.curr_channel = 1
        self.nick = None
        
        self.base = "http://{}:{}".format(ip, port)
        self.lasttime = datetime.now()
        self.update_nick(nick)
        
    def update_nick(self, new_nick):
        content = json.dumps({'old_nick': self.nick, 'new_nick': new_nick})
        r = requests.post(self.base + "/changenick", data=content)
        if r.ok:
            self.nick = new_nick
        else:
            new_nick = uuid1().hex
            print "Assigning random nick: ", new_nick
            self.update_nick(new_nick)
    
    def pull(self):
        content = json.dumps({'channel': self.curr_channel, 'timestamp': self.lasttime, 'nick': self.nick}, cls=DateEncoder)
        r = requests.get(self.base + "/message/pull", data=content)
        
        if r.ok:
            self.pretty_printer(x for x in json.loads(r.text, object_hook=dateobj))
    
    def push(self, message):
        self.lasttime = datetime.now()
        msg = {'timestamp': self.lasttime,
                'content': message,
                'channel': self.curr_channel,
                'nick': self.nick}
        
        r = requests.post(self.base + "/message/push", data=json.dumps(msg, cls=DateEncoder))
        
    def pretty_printer(self, msglist):
        for m in msglist:
            print "{} {}: {}".format(m['timestamp'], m['nick'], m['content'])
            
    
def main_loop(ip, port, nick):
    session = Session(ip, port, nick)
    prompt = "{} >>> "
    while True:
        msg = raw_input(prompt.format(datetime.now()))
        
        #check for special commands
        if msg.startswith('//'):
            #parse a command
            command = msg[2:].split()
            if command[0] == 'quit':
                print "Goodbye."
                break
            if len(command) == 2 and all(command):
                if command[0] == 'join':
                    session.curr_channel = int(command[1])
                elif command[0] == 'nick':
                    session.update_nick(command[1])
                    
                
            print "You are {} listening on channel {}".format(session.nick, session.curr_channel)
            continue
        
        session.push(msg)
        session.pull()
    
if __name__ == "__main__":
    ip = raw_input("Server IP: ")
    port = input("Server Port: ")
    nick = raw_input("Nickname: ")
    main_loop(ip, port, nick)
    