:: filter cmake project 
dir /s /b *.tars  > 1.win.txt
dir /s /b *.cpp |grepw /V thirdparty  |grepw /V vs_build |grepw /V t0 >> 1.win.txt 
dir /s /b *.h  |grepw /V thirdparty  |grepw /V vs_build |grepw /V t0 >> 1.win.txt

dir /s /b /A-D thirdparty\*.h |grepw "tc_common.h tc_ex.h tc_encoder.h tc_file.h tc_json.h tc_logger.h tc_md5.h tc_mysql.h tc_option.h tc_platform.h tc_port.h tc_thread.h tc_thread_pool.h tc_thread_rwlock.h tc_thread_queue.h tc_sqlite.h" >> 1.win.txt 

dir /s /b CMakeLists.txt |grepw /V "thirdparty sqlite protocol classifer so common" >> 1.win.txt  



:: >> if filter vs sln:
:: tar cvzf 1.tgz ttbb.cpp ttbb.sln ttbb.vcxproj ttbb.vcxproj.filters ttbb.vcxproj.user 
