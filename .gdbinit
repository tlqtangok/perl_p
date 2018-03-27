
define tc
	python import time
	python starttime=time.time()
	continue
	python print (time.time()-starttime)
end

define tn
	python import time
	python starttime=time.time()
	next
	python print (time.time()-starttime)
end

define tf
	python import time
	python starttime=time.time()
	finish	
	python print (time.time()-starttime)
end

define tcontinue
	python import time
	python starttime=time.time()
	continue
	python print (time.time()-starttime)
end

define tnext
	python import time
	python starttime=time.time()
	next
	python print (time.time()-starttime)
end

define tfinish
	python import time
	python starttime=time.time()
	finish	
	python print (time.time()-starttime)
end
