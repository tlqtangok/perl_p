@echo off
:: Wrote by Jidor, at 2015-6-30 20:55¡¡
setlocal enabledelayedexpansion
set _txt_=%1
if "!_txt_!"=="" (
		echo cat filename 3-6 OFF
		::pause
		goto :EOF
	       )
set _line_start_=%2
:: control the line number
set _set_num_on_=%3
::echo !_set_num_on_!
:: echo ------
if "!_set_num_on_!"=="" (
		set _set_num_on_=OFF
		) 
if "!_set_num_on_!"=="ON" (
		set _set_num_on_=ON
		) else (
			set _set_num_on_=OFF
		       )
	::echo %_set_num_on_%
	::set line_from_to=
	del /f /q _tmp_.txt  2>nul
	echo %_line_start_%|findstr /m "\-" > nul && echo YES> _tmp_.txt || echo NO> _tmp_.txt
	set /p _var_of_YES_NO_=<_tmp_.txt&& del _tmp_.txt


	::echo %_var_of_YES_NO_%
	find /c /v "" %_txt_% | find ": " |repl "^.*\: " "" X > _tmp_.txt 
	set /p _txt_len_=<_tmp_.txt && del _tmp_.txt
	::	echo !_txt_len_!



	if "%_var_of_YES_NO_%"=="NO" (
		::	echo only one num
			set _start_line_=%_line_start_%
			set _to_line_=!_txt_len_!
			if "!_start_line_!"=="" (
				set _start_line_=0
				)


			)



	if "%_var_of_YES_NO_%"=="YES " (
			echo %_line_start_% |repl "\-.*$" "" X>_tmp_.txt
			set /p _start_line_=<_tmp_.txt && del _tmp_.txt
			echo %_line_start_% |repl "^.*\-" "" X|repl " " "" X>_tmp_.txt
			set /p _to_line_=<_tmp_.txt&& del _tmp_.txt
			::	echo !_to_line_!
			if "!_to_line_!"=="" (
				set _to_line_=!_txt_len_!
				)
			if !_start_line_! GTR !_txt_len_! (
				echo wrong _start_line_ num: _!_start_line_!_
				) 
			if !_to_line_! GTR !_txt_len_! (
				echo set _to_line_ num: !_txt_len_!
				)
			)
	:: end YES

	set _cnt_=1

	for /F "delims=" %%i in (%_txt_%) do (
			if !_cnt_! GEQ !_start_line_! (
				if !_cnt_! LEQ !_to_line_! (
					if "!_set_num_on_!"=="ON" (
						echo !_cnt_!: %%i

						) 
					if "!_set_num_on_!"=="OFF" (
						echo %%i
						)
					)
				)
			set /a _cnt_+=1
			)
	:EOF
	 del /f /q _tmp_.txt  2>nul
	 endlocal 
	 @echo on 
