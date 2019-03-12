#!python3
# written by jd, 2019.03.10
# called by wx.js for ls.

import os
import sys



print ("hello world from ls") 
print ( os.popen("echo " + sys.argv[-1] + "|base64 -d " ).read() )
