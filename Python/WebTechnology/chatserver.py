from flask import Flask, request, Response, redirect, url_for
app = Flask(__name__)

from chatsession import Session
import json
import sys

from datetime import datetime

session = Session()

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

@app.route('/')
def index():
    return '''This is a simple messaging program'''

@app.route('/changenick', methods=["POST"])
def change_nick():
    msg = json.loads(request.data)
    try:
        session.change_nick(msg['old_nick'], msg['new_nick'])
        return Response(status=200)
    except KeyError:
        return Response(status=500)

@app.route('/channels/<channel>')
def view_channel(channel):
    if channel == '0':
        print "returning all"
        return '<br>'.join(str(x) for x in session.sessionlog)
    
    channels = set([0, int(channel)])
    return '<br>'.join('{}: {}'.format((x['nick'], x['content']) for x in session.sessionlog if x['channel'] in channels))

@app.route('/channel/create', methods=["POST"])
def create_channel():
    msg = json.loads(request.data)
    try:
        session.register_channel(msg['channel'])
        return Response(status_code=200, status='OK')
    except:
        return Response(status_code=500, status='SERVER ERROR')
    
@app.route('/channel')
def get_channels():
    r = Response(status_code=200, status="OK")
    r.text = json.dumps(session.channels)
    return r

@app.route('/message/push', methods=["POST"])
def send_msg():
    msg = json.loads(request.data, object_hook=dateobj)
    print request.data
    ch = msg['channel']
    print "Trying to post to channel", ch
    try:
        session.log_message(msg)
        print "Message logged on channel", ch, msg
        return request.data
    except ValueError:
        print "No such channel exists!"
        return Response(status_code=404, status="NOT FOUND")
    
@app.route('/message/pull')
def get_msg():
    req = json.loads(request.data, object_hook=dateobj)
    print req
    msgs = session.retrieve_new(req['nick'], req['timestamp'], req['channel'])    
    return json.dumps(msgs, cls=DateEncoder)
                
            
    

if __name__ == "__main__":
    host, port = sys.argv[1:3]
    app.run(host=host, port=int(port), threaded=True, debug=True)