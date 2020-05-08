#!python
# readme 
# written by Jidor Tang <tlqtangok@126.com> , 2018-12-15

# a tiny command line chat tools based redis, python3
# dependency: redis , 
#   . run "pip3 install redis" to install 

# steps
#   . login use an id
#   . CTRL+C to send msg 

import redis
import time
import os
from multiprocessing import Process

import signal
import datetime
import pickle
import sys

### def list ###
global flag_check_msg 
flag_check_msg = [1]  # default check msg

# utils_ begin 
def get_timestamp():
    dt = datetime.datetime.now()
    return dt

def chomp(id_str):
    return id_str.strip()
# utils_ end

def connect_r(host, port=6379):
    return redis.Redis(host=host, port=port, db=0)

def signal_handler_sys_exit(sig, frame):
    sys.exit(1)

def signal_handler_send_msg(sig, frame):
    flag_check_msg[0] = 0

def login_as():
    id_ = input("- login as: ")
    id_ = chomp(id_)
    assert( id_ != "")
    return id_

def create_msg_from_whom(id_):
    timestamp = get_timestamp()
    from_whom = id_
    to_whom = "NULL"
    msg_body = ""
    to_whom = input("- send msg to whom? ")
    to_whom = chomp(to_whom)

    print("- input your msg content, use END to end input \nBEGIN\n")
    
    e_l = ""
    while e_l != "END":
        msg_body += e_l + "\n"
        e_l = input("> ")

    msg_pkg = {
            "timestamp": timestamp,
            "from":from_whom,
            "to":to_whom,
            "msg_body":msg_body
    }

    msg_pickle_string = pickle.dumps(msg_pkg)
    return [msg_pkg, msg_pickle_string]

def pretty_msg_pkg(msg_pkg):
    ret_str = ""
    my_id_ = msg_pkg["to"]
    from_ = msg_pkg["from"]
    msg_body = msg_pkg["msg_body"]

    timestamp = msg_pkg["timestamp"]
    timestring = timestamp.strftime("%Y-%m-%d %H:%M")
    ret_str = "\n------ " + timestring + " msg from " + from_ +   " ------" + msg_body + "\n"
    return ret_str

def check_msg_to_me(con, my_id_):
    MSG_PREFIX = "MSG_TO_"
    msg_pickle_string = con.lpop(MSG_PREFIX + my_id_)
    if msg_pickle_string != None:
        msg_pkg = pickle.loads(msg_pickle_string)
        print(pretty_msg_pkg(msg_pkg))
    return msg_pickle_string

def send_msg_to_whom(con, msg_pickle_string, msg_pkg):
    MSG_PREFIX = "MSG_TO_"
    con.rpush(MSG_PREFIX + msg_pkg["to"] , msg_pickle_string)
    return "- send msg from " + msg_pkg["from"] + " to " + msg_pkg["to"] + ", done!"

### end def ###

if __name__ == "__main__":

    if len(sys.argv) == 2 and (sys.argv[1] == "-V" or sys.argv[1] == "-v"):
        #print  (sys.argv[1])
        print ("cct\tCommand-line Chat Tools\nversion 1.0\n\nwritten by Jidor Tang<tlqtangok@126.com> at 2018-12-15")
        sys.exit(1)
         
    signal.signal(signal.SIGINT, signal_handler_send_msg)
    #signal.signal(signal.SIGQUIT, signal_handler_sys_exit)

    host = "e.e.a.b.com"
    host = ''.join(list(reversed("m o c . b a l e w e l".replace(" ", ""))))
    port = 6379

    con = connect_r(host)

    id_ = login_as()
    print ("login as " + id_)

    cnt_check = 0
    print("- press [CTRL + C] to send msg !")

    while True:
        ck_status = None
        if flag_check_msg[0]:
            ck_status = check_msg_to_me(con, id_)
        else:
            flag_check_msg[0] = 1
            cnt_check = 0
            s_or_r = input("\n- do you want to send msg Y(YES) | N(NO)?: ") 
            s_or_r = chomp(s_or_r)

            if s_or_r == "YES" or s_or_r == "Y" or s_or_r == "yes" or s_or_r == "y":
                [msg_pkg, msg_pickle_string] = create_msg_from_whom(id_)
                send_res = send_msg_to_whom(con, msg_pickle_string, msg_pkg)
                print (send_res)

        if ck_status == None and cnt_check % 25 == 0: 
            print (".", flush=True,end="")
        
        cnt_check += 1
        if cnt_check == 0.5 * 60 * 60:
            sys.exit(1)

        time.sleep(1)
    con.connection_pool.disconnect()

#1
