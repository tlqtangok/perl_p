#!/bin/bash 
set -x

SCRIPT_WINPATH=`cygpath --windows --absolute "$1"`
EXPLORER_CYGPATH=`which explorer`
EXPLORER_WINPATH=`cygpath --windows --absolute "${EXPLORER_CYGPATH}"`

cmd /C start "clean shell" /I "${EXPLORER_WINPATH}" "${SCRIPT_WINPATH}"

