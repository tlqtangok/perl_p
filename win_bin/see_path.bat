@echo off
pushd %cd%
echo %path% |repl ";" ";\n" X
popd