#!python3
# written by jd, 2019.03.10
# called by wx.js for ls.

from pathlib import Path
import os
import sys

######################
def ck_args_and_deps(fn_req):
    assert(sys.argv[-1] == fn_req)
    fn_req_fp = Path(fn_req)
    assert(fn_req_fp.is_file())


def get_req_str(fn_req):
    fc = os.popen( "cat " + fn_req).read()
    return fc

def proc_fc(fc):
    return fc

def ret_to_wx_js(fc):
    print(fc, end="")
#######################


if __name__ == "__main__":
    fn_req = "ls.wx.req.txt" 
    ck_args_and_deps(fn_req)

    fc = get_req_str(fn_req)

    proc_fc(fc)

    ret_to_wx_js(fc)



