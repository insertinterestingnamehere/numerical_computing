import json
import datetime

# This is _one_ of several possible solutions.
class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return str(obj)
        return json.JSONEncoder.default(self, obj)

def date_decoder(dct):
    def dedate(s):
        #ignore the quote marks (we have a string inside a string)
        parts = s[1:-1].split('.')
        r = datetime.datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
        if parts[1]:
            r.replace(microsecond=int(parts[1]))
        return r
    
    #try to decode any value that looks like a date
    for i, k in dct.iteritems():
        try:
            dct[i] = dedate(k)
        except:
            continue
    return dct
