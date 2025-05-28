# in taskschd.msc
#
# powershell.exe -NoProfile -ExecutionPolicy Bypass -File "d:\jd\pro\ngrok_client_win\start_ngrok.ps1"
sleep 1;

echo "Starting rustdesk...";
& "d:\jd\pro\rustdesk\start_rustdesk.bat" -RunAsJob;
echo "running rustdesk...";

sleep 1;
echo "Starting ngrok...";
& "D:\jd\pro\ngrok_client_win\start_ngrok.bat" -RunAsJob;  
echo "running ngrok...";

sleep 1;

pause;


# 以作业方式运行 test1.bat 和 test2.bat
# & "D:\path\to\test1.bat" -RunAsJob
# & "D:\path\to\test2.bat" -RunAsJob
