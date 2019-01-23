#!node

var Promise = require("bluebird");
var child_process = require("child_process");
var fs = require("fs");
var print = console.log;
var assert = require("assert");

require("./lib/dep.js"); //require("./lib/dep.js");    // utils => R(5) print() sleep_sync_sec() count_arc_info() 






main();


// ### sub list ###
// main_
function main() {

	Promise.promisifyAll(fs);
	Promise.promisifyAll(exec_asyn_ms);

	var td_all = [];
	for (var i = 0; i < 3; ++i) {
		//td_all.push(fs.writeFileAsync("file-" + i + ".txt", "", "utf-8"));
		//td_all.push(exec_asyn_ms(print, 1*1000, `sleep ${i}, then print data`)); 
	}

	for (var i = 0; i < 3; ++i) {
		//files.push(print_("print data " + i )); 
	}

	var xx = "XXX";


	if(1)
	{

		//var a = `0123456789name="---------0440a19f2a262529
		var buf = Buffer.from([0,0,0,0, 0x0a,0x0a,0x0d,0x0a, 0,0,0,0]);
for( let e_b of buf)
{
	print(e_b); 
}

var t = {
	a:"a"
};
		assert(0==1); 


		var FN_SUFFIX = `name="`;

		//var buf_FN_SUFFIX = Buffer.from(FN_SUFFIX); 
		var quato_FN = `"`;
		var two_new_line = Buffer.from([0x0a,0x0d,0x0a]); 


		print(two_new_line); 
		var start = 0; 

		var idx_fn_start = 0; 
		var idx_fn_end = 0; 
		var buf_fn; 

		idx_fn_start = buf.indexOf(FN_SUFFIX, start) + FN_SUFFIX.length;
		idx_fn_end = buf.indexOf(quato_FN, idx_fn_start); 
		buf_fn = buf.slice(idx_fn_start, idx_fn_end); 
		start += idx_fn_end; 
		var fn_rename = buf_fn.toString() ; 

		idx_fn_start = buf.indexOf(FN_SUFFIX, start) + FN_SUFFIX.length;
		idx_fn_end = buf.indexOf(quato_FN, idx_fn_start); 
		buf_fn = buf.slice(idx_fn_start, idx_fn_end); 
		start += idx_fn_end; 
		var fn = buf_fn.toString(); 


		print("___" + fn_rename + "___"); 
		print("___" + fn + "___"); 




		idx_fn_start = buf.indexOf(two_new_line.toString(), start) + two_new_line.length;
		start +=  idx_fn_start; 

		idx_fn_end = buf.indexOf(two_new_line.toString(), start);

		var buf_fc = buf.slice(idx_fn_start, idx_fn_end); 

		print("___" + buf_fc.toString() + "___"); 


	}


	if(0)
	{
		Promise.using(print_0_no_args(), print_1_no_args(), (d0, d1) =>{
			print (d0); 
			print (d1); 
		});

	}

	if (0) 
	{
		var arr_fn = ["file1.txt", "file2.txt", "file3.txt"];

		Promise.map(arr_fn, e_fn => {
			return fs.readFileAsync(e_fn, "utf8");
		})
		.then(arr_fc => {
			print(`list is ${arr_fc}`);
		});

		Promise.each(arr_fn, e_fn => {
			return fs.readFileAsync(e_fn)
				.then(e_fc_bytes => {
					print(e_fc_bytes.toString());
				});
		})
		.then(arr_fn_ => {
			print(arr_fn_);
		});



		Promise.map(arr_fn, e_fn => {
			return fs.readFileAsync(e_fn, "utf8");
		})
		.each(e_fc => {
			print(e_fc);
		});



		Promise.filter([print_0_no_args(), print_1_no_args(), print_0_no_args()], e => {
			return m_(e, /1/gi);
		})
		.each(
				e_match => {
					print(e_match);
				});

	}

	if (0) {
		Promise.all([print_0(xx), print_1(xx)])
			.then(data => {
				print(`list data is ${data}`);
			});

		Promise.some([print_0(xx), print_1(xx)], 1)
			.then(data => {
				print(data);
			});

		Promise.any([print_0(xx), print_1(xx)])
			.then(data => {
				print(data);
			});


	}


	if (0) {
		var fn_db_ = "file1.txt";
		Promise
			.reduce([fn_db_, fn_db_, fn_db_], // 1,2,3
					(sum, fn) => {
						return fs.readFileAsync(fn, 'utf8')
							.then(fc => {
								var int_a = parseInt(fc, 10);
								//var int_a = fc;
								return `${sum}${int_a}`;
							});
					}, "FC:")
		.then((sum) => {
			print(sum);
			print("- OK");
		});
	}



	if (0) {
		Promise.join(
				print_0(xx), print_1(xx), print_0(xx),
				(a0, a1, a2) => {
					print("-a0 " + a0);
					print("-a1 " + a1);
					print("-a2 " + a2);
				});
	}


	if (0) {
		Promise.resolve(print_0("e"))
			.then((data) => {

				print("- data " + data);
			});
	}


	if (0) {
		Promise
			.delay(0)

			.then(() => {
				setTimeout((data) => {
					print(data);
				}, 1 * 1000, 1);
			})

		.delay(0.9 * 1000)

			.then(
					() => {
						print(xx + xx);
						return xx + xx;
					});
	}


	//print(td_all[0]);	
	if (0) {
		Promise.all(td_all).then(function() {
			console.log("all the files were created");
		});
	}

	//print(td_all[0]);	


	if (0) {
		Promise.some(files, 2).then(function(first, second) {
			console.log("any " + first + ":" + second);
		});
	}
};

function print_0(e) {
    print(e + "0" + "- in print_0");
    return e + "0";
}

function print_0_no_args() {
    print("e" + "0" + "- in print_0");
    return "e " + "0";
}

function print_1_no_args() {
    print("e" + "1" + "- in print_1");
    return "e " + "1";
}

function print_1(e) {
    print(e + "1" + "- in print_1");
    return e + "1";
}


function split_(re, id_text) {
    return id_text.split(re);
}

function exec_asyn_ms(cb, time_to_sleep_i_ms, data) {
    return setTimeout(cb, time_to_sleep_i_ms, data);
}

function process_e_(e) {

    child_process.exec(`ls -al ${e}`, (c, d, err) => {
        print(`ls -al ${e}`);
        print(d);

        if (err) {
            print("- ERROR msg is: \n" + err + "- end ERROR msg");
        }
        //return d;
    });
};

function data_to_e_and_process_e(data) {
    split_("\n", data).forEach((e) => {
        if (e != "") {
            process_e_(e);
        }
    });
    return "_in_then_1()";
};

function print_put_data(data) {
    print(data);
    return "_in_then_0()\n" + data;
};

function m_(id_text, re) {
    return id_text.match(re);
}
