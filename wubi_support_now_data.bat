::cp_to_bak %userprofile%\AppData\Roaming\Microsoft\InputMethod\Chs\ChsWubiEUDPv1.lex
:: manully set up 2 shortcuts as following:
    ::date:
    ::_yyyy_-_MM_-_dd_
    ::now: 
    ::_yyyy_-_MM_-_dd_ _HH_:_mm_:_ss_

set src=%userprofile%\AppData\Roaming\Microsoft\InputMethod\Chs\ChsWubiEUDPv1.lex
set dst=%t%\ChsWubiEUDPv1.lex.txt
xxd %src%  %dst%
perl -p -i.bak -e "s/5f00/2500/g" %dst%
xxd -r %dst% %src%
cp %dst% %src%
del %dst% %dst%.bak
