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


define b2
	b $arg0
	b $arg1
end

define b3
	b $arg0
	b $arg1
	b $arg2
end
define b4
	b $arg0
	b $arg1
	b $arg2
	b $arg3
end

define bm
	if $argc == 2
		b2 $arg0 $arg1
	end
	if $argc == 3
		b3 $arg0 $arg1 $arg2
	end
	if $argc == 4
		b4 $arg0 $arg1 $arg2 $arg3
	end
	if $argc == 5
		b4 $arg0 $arg1 $arg2 $arg3
		b $arg4
	end
	if $argc == 6
		b4 $arg0 $arg1 $arg2 $arg3
		b $arg4
		b $arg5
	end
	if $argc == 7
		b4 $arg0 $arg1 $arg2 $arg3
		b $arg4
		b $arg5
		b $arg6
	end
end


define tcontinue
	tc
end

define tnext
	tn
end

define tfinish
	tf
end

define bsave
	save breakpoints ~/.breakpoints
end

define brestore
	source ~/.breakpoints
end
