redis-server --service-install  redis.windows-service.conf --service-name redis_service_jd_10240 --loglevel notice 

redis-server --service-start --service-name  redis_service_jd_10240
:: redis-server --service-stop  --service-name  redis_service_jd_10240
:: redis-server --service-uninstall   --service-name  redis_service_jd_10240
::
:: .\redis-cli -p 10240 -h 192.168.100.161   keys "*"
