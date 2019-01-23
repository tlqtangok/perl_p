var express = require('express');
var https = require('https');
var http = require('http');
var fs = require('fs');
var app = express();
var assert = require("assert"); 

var execSync_ = require("child_process").execSync;
var exec_ = require("child_process").exec;
var print = console.log ;
//

var bodyParser = require("body-parser");
//app.use(bodyParser.json()); 
//app.use(bodyParser.raw()); 
//app.use(bodyParser.raw({ type: 'application/vnd.custom-type' }))

//app.use(bodyParser.urlencoded({extended: true})); 


var port = 10203;

var options = 
	{
		//key: fs.readFileSync('/path/to/key.pem'),
		//cert: fs.readFileSync('/path/to/cert.pem')
	};

var m_=( id_text, re )=>{
	return id_text.match(re); 
}

// respond with "hello world" when a GET request is made to the homepage
app.get('/*', function (req, res) 
	{

		console.log("___req___"); 
		//console.log(req); 

		console.log(req.method); 
		console.log(req.headers.host); 
		console.log(req.url); 

		var b64_url = Buffer.from(req.url).toString("base64")

		//var cmd_out = execSync_("perl test.PL " + b64_url); 

		var test_PL = "NULL"; 

		if(m_(req.url, /test_long/gi))
		{	
			test_PL = "test_long.PL"; 
		}
		else
		{
			test_PL = "test.PL"; 

		}

		exec_(`perl ${test_PL} `+ b64_url, (error, stdout, stderr) => {
			if (error) {
				console.error(`exec error: ${error}`);
				return;
			}
			console.log(`stdout: ${stdout}`);
			//console.log(`stderr: ${stderr}`);
			//res.status(404).json({error: "Error msg"}); 
			console.log("___res___"); 
			//console.log(res); 
			console.log(res.statusCode); 
			res.send('get method, hello world' );
			//res.send('' + cmd_out);
		});



	});

// POST method route
//var bodyParser = require("body-parser");
app.use(bodyParser.json({ limit: '4gb' }));
app.use(bodyParser.raw({ type: 'audio_/jd_', limit: '4gb' }));


app.post ('/*',function (req, res) 
	{

		var flag = get_option_of_curl_F_T_JSON(); 

		print ("___" + flag + "___");



		//app.use(bodyParser.raw({ type: 'audio_/jd_', limit: '4gb' }));
		//console.log("RECIEVED AUDIO TO EXTRACT INDICATORS: ", req.body);


		if ("-json" == flag)
		{
			`
			curl -X POST -H "Content-Type: application/json" --data '{"username":"xyz","password":"xyz"}' "http://localhost:10203/___path___" 
			`;
			console.log(typeof(req.body)); // object 
			print(req.body); 
			print("- json flag");
		}



		if ( "-F" == flag )
		{
			`
			curl -v -X POST -H "Content-Type: audio_/jd_"   -F "1.txt.rename=@1.txt"  "http://localhost:10203/api/login"
			`;
			var dict_req = get_fn_fc(req.body);   // if use -F
			print(dict_req); 

			execSync_("rm txt.txt"); 
			fs.writeFileSync("txt.txt", dict_req.buf_fc, 'binary'); 
			fs.writeFileSync(dict_req.fn_rename, dict_req.buf_fc, 'binary'); 
			print (execSync_("cksum txt.txt").toString()); 
			print("-F flag");
		}


		if ("-T or --data-binary" == flag )
		{
			`
			curl -X POST -H "Content-Type: audio_/jd_" -T "{$t/1.tar.gz,$t/1.txt}" localhost:10203/api/test
			curl -v -# -X POST -H "Content-Type: audio_/jd_" --data-binary @'1.txt' "localhost:10203/___path___"
			`;
			execSync_("rm txt.txt"); 
			fs.writeFileSync("txt.txt", req.body, 'binary');  // if use -T
			print (execSync_("cksum txt.txt").toString()); 
			print("-T or --data-binary flag");

		}



		//var sh_sh =  req.body.toString();

		//print(sh_sh); 
		//print(req.body.length)
		//print(sh_sh.length)

		//fs.writeFileSync("txt.txt", Buffer.from(req.body)); 

		//console.log("___req___"); 
		//console.log(req.body); 
		//console.log(req.body.length); 

		//print(req.buffer().length());

		//console.log(typeof(req.body.jd)); 

		//const id_b =  Buffer.from(req.body.jd);
		//console.log(id_b); 

		//fs.writeFileSync ( "1.tar.gz",  id_b, 'binary');

		//console.log("___res___"); 
		//console.log(res); 



		res.send('POST request to the homepage')

		//res.send(req.body.jd);

		if (0)
		{
			`
			# OK, json 
			curl -X POST -H "Content-Type: application/json" --data '{"username":"xyz","password":"xyz"}' "http://localhost:10203/___path___" 


			curl -v -X POST -H "Content-Type: audio_/jd_"   -F "1.txt.rename=@1.txt"  "http://localhost:10203/api/login"


			# -T filename , and usage progress -#
			curl -X POST -H "Content-Type: audio_/jd_" -T "{$t/1.tar.gz,$t/1.tar.gz}" localhost:10203/api/test
			curl -X POST -H "Content-Type: audio_/jd_" -T '[1-2].tar.gz' localhost:10203/api/test
			curl -X POST -H "Content-Type: audio_/jd_" -T '1.tar.gz' localhost:10203/api/test

			# OK, binary file , also can use $t/bin.bin
			curl -v -# -X POST -H "Content-Type: audio_/jd_" --data-binary @"$t/1.tar.gz" "localhost:10203/___path___"

			`;

		}

	}
);



//http.createServer(app).listen(80);

app.all('/secret', function (req, res, next) 
	{
		console.log('Accessing the secret section ...')
		next() // pass control to the next handler
	});

app.listen(port); 

//https.createServer(options, app).listen(443);

console.log('Server started! At http://localhost:' + port);



function get_fn_fc(buf)
{
	assert(typeof(buf) == typeof(Buffer.from("")));


	var FN_SUFFIX = `name="`;

	//var buf_FN_SUFFIX = Buffer.from(FN_SUFFIX);
	var quato_FN = `"`;
	var two_new_line = Buffer.from([0x0a,0x0d,0x0a]);
	var two_new_line_end = Buffer.from([0x0d,0x0a,0x2d]);


	//assert(0==1); 

	print(two_new_line);
	var start = 0;

	var idx_fn_start = 0;
	var idx_fn_end = 0;
	var buf_fn;

	idx_fn_start = buf.indexOf(FN_SUFFIX, start) + FN_SUFFIX.length;
	idx_fn_end = buf.indexOf(quato_FN, idx_fn_start);
	buf_fn = buf.slice(idx_fn_start, idx_fn_end);
	start = idx_fn_end;
	var fn_rename = buf_fn.toString() ;

	idx_fn_start = buf.indexOf(FN_SUFFIX, start) + FN_SUFFIX.length;
	idx_fn_end = buf.indexOf(quato_FN, idx_fn_start);
	buf_fn = buf.slice(idx_fn_start, idx_fn_end);
	start = idx_fn_end;
	var fn = buf_fn.toString();


	//print("___" + fn_rename + "___");
	//print("___" + fn + "___");

	//var idx_0 = buf.indexOf(two_new_line, start) + two_new_line.length; 

	//var idx_1 = buf.indexOf(two_new_line, idx_0); 

	//var left_1 = buf.slice(idx_0,idx_1).toString(); 

	idx_fn_start = buf.indexOf(two_new_line, start) + two_new_line.length;
	start =  idx_fn_start;


	idx_fn_end = buf.indexOf(two_new_line_end, start);

	var buf_fc = buf.slice(idx_fn_start, idx_fn_end);

	//print("___" + buf_fc.toString() + "___");


	return {fn_rename: fn_rename, fn: fn, buf_fc: buf_fc};

}

function get_option_of_curl_F_T_JSON()
{

	`
		*********************
		flag.txt
		*********************
		-F
		-json
		-T or --data-binary
		`;

	var this_fn = __filename; 	
	var flag = fs.readFileSync(this_fn).toString();
	flag = flag.split("*********************\n")[2].split("\n")[0].replace(/^.*?\-/m,"-");
	//print("___"+flag+"___"); 

	return flag; 
}


`
# OK, json 
curl -X POST -H "Content-Type: application/json" --data '{"username":"xyz","password":"xyz"}' "http://localhost:10203/___path___" 


curl -v -X POST -H "Content-Type: audio_/jd_"   -F "1.txt.rename=@1.txt"  "http://localhost:10203/api/login"


# -T filename , and usage progress -#
curl -X POST -H "Content-Type: audio_/jd_" -T "{$t/1.tar.gz,$t/1.tar.gz}" localhost:10203/api/test
curl -X POST -H "Content-Type: audio_/jd_" -T '[1-2].tar.gz' localhost:10203/api/test
curl -X POST -H "Content-Type: audio_/jd_" -T '1.tar.gz' localhost:10203/api/test

# OK, binary file , also can use $t/bin.bin
curl -v -# -X POST -H "Content-Type: audio_/jd_" --data-binary @"$t/1.tar.gz" "localhost:10203/___path___"
`;

`
#!sh
# sh.sh , run : sh sh.sh file_path.txt
export fn=$1

export localhost_port="172.16.29.10:10203"

cksum $fn
curl -X POST -H "Content-Type: audio_/jd_" -T $fn $localhost_port/sfksdfjs
`;


