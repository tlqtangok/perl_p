import yagmail
import sys
from pathlib import Path
import re

# pip3 install yagmail

def s_(id_text, pat_old, pat_new):
    return re.sub(pat_old,  pat_new, id_text)

def get_subject(content_text):
    list_ret = []
    for i in content_text:
        if i!="\n":
            list_ret.append(i)
    return "".join(list_ret)




if __name__ == "__main__":
    ### global var ###
    magic_number = "z"
    ##################

    argc_r0 = 1 
    _at_ = '@';
    send_type = "text"
    content_text = "NULL" 
    img_suffix = [".jpg",".jpeg", ".png", ".bmp"]
    password='jhdlubsemxhjbedc'

    user_mail_addr = "9"+magic_number+"837"+magic_number+"829"+_at_+"qq.com" 
    to_mail_addr = "tlqtangok"+_at_+"126.com"

    id_yagmail = yagmail.SMTP(user = user_mail_addr, password=password, host = 'smtp.qq.com')

    #print (len(sys.argv))
    #print (sys.argv)
    if len(sys.argv) <= 0 + argc_r0:
        print ( "- usage :\n\tpython3 ygmail.py text filename" )
        # python3 ygmail.py content  1.txt
        exit(1)

    content_text = sys.argv[0 + argc_r0]


    if len(sys.argv) >= 2 + argc_r0:
        fn = sys.argv[1 + argc_r0]
        fn_ = Path(fn)
        if not fn_.is_file():
            print ("- "+ fn + " is not exists !")
            assert(0==1)
        flag_img = 0
        for e_img in img_suffix:
            if fn.endswith(e_img):
                flag_img = 1
                break

        if flag_img:
            send_type = "img"
        else:
            send_type = "attachment"
    

    #print(send_type)
    #print(content_text)
    
    if send_type == "text":
        contents = [ content_text ]

    if send_type == "img":    
        contents = [ content_text, yagmail.inline(sys.argv[1+argc_r0]) ]

    if send_type == "attachment":    
        contents = [ content_text, sys.argv[1+argc_r0] ]

    sub_content = "jd_send_by_ygmail: " 
    content_text_no_line_LF = get_subject(content_text)
    MAX_LEN = 25
    if len(content_text_no_line_LF)> MAX_LEN:
        sub_content = sub_content + content_text_no_line_LF[0:MAX_LEN] + " ..."
    else:
        sub_content = sub_content + content_text_no_line_LF



    print("- from\t: " + user_mail_addr)
    print("- to\t: " + to_mail_addr)
    print("- content_text\t: " + "\n" + "======")
    print( content_text )

    if send_type == "img":
        print ("-")
        print ("- img: " + sys.argv[1+argc_r0])

    if send_type == "attachment":
        print ("-")
        print ("- attachment: " + sys.argv[1+argc_r0])
    print("======")


    id_yagmail.send(to = to_mail_addr, subject = sub_content, contents = contents)


    print("\n"+"- OK, done")
