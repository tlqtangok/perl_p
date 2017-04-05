 echo $1 |perl -e ' $_=<>; s|\/cygdrive\/([c-z])\/|\1\:\\|i ; s|\/|\\|g ; print ; '
