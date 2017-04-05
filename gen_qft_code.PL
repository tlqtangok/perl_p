#!perl 
use strict; 
use Data::Dumper; 

my $qft_file_name = @ARGV[0]; 
my $function_pattern = @ARGV[1]; 

if ( ( ! ( $qft_file_name  =~ m/\.qft/ ) ) && @ARGV == 0 ){
	print "- usage:\n"; 
	print "\tperl " .__FILE__ . " xxx.qft fun_pattern \n"; 
	exit ; 
}



my @arr_file = `cat $qft_file_name`;
my @proc_list = &get_proc_name_list( \@arr_file );



#print "@proc_list"; 


@proc_list = grep !m/______/, @proc_list; 
my %hash_ = (); 
@proc_list = grep { ++$hash_{$_} < 2 } @proc_list;


if ( @ARGV == 1 ){
	print "### the function list ###\n"; 
	my $str_content = join "\n ", @proc_list; 
	print " ", $str_content, "\n" ; 
	exit ; 

}



my @fun_pattern_list = @ARGV[1..@ARGV-1]; 


my @function_list = &from_pattern_to_precise(\@fun_pattern_list, \@proc_list); 

my $i = 0; 
for ($i = 0; $i<@function_list; $i++ ) {
	&grep_by_each_function_name(\@arr_file,  @function_list[$i] ); 
}



#print &get_call_statement($proc_name, \@arr_variables,$ret_val); 

sub get_procedure_name($proc_string){
	my ($proc_string) = (@_[0]); 
	$proc_string =~ s/.*name\=\"(.*)\".*$/\1/; 
	return $proc_string; 

}

sub detach_var_state($var_string){
	my ($var_string ) = (@_[0]);
	chomp( $var_string); 
#print $var_string ; 
	$var_string =~ s/^.*name\=\"(.*)\"\>(.*)\<.*$/\1\:\2/;
#print $var_string; 
	my @detached_var = split m/\:/, $var_string; 
	return @detached_var; 



}

sub get_ret_val($ret_val_statement){

	my ($ret_val_statement) = (@_[0] ); 
	chomp( $ret_val_statement); 
	$ret_val_statement =~ s/[\$\(\)]//g; 
	$ret_val_statement =~ s/^.*retval\=\"(.*)\".*$/\1/; 
	return $ret_val_statement; 

}

sub get_proc_name_list ( \@arr_file){
	my ($arr_file) = (@_[0]); 
	my $i=0; 

	my @proc_list = (); 
	for ( $i=0; $i<@$arr_file; $i++){
		if (@$arr_file[$i] =~ m/\<Procedure id/ ){
			#	push @proc_list, @$arr_file[$i].@$arr_file[$i+1] 
			if (! (@$arr_file[$i+1] =~ m/\</)){
				my $pre= @$arr_file[$i] ; 
				chomp($pre); 	
				push @proc_list, $pre.@$arr_file[$i+1] ;
			}
			else{	
				push @proc_list, @$arr_file[$i] ; 
			}
		}

	}

#	print @proc_list ; 

	map{
	chomp;
	s/^.*?name=\"(.*?)\"\>.*$/\1/m;
	s/\".*$//;
	}@proc_list ; 

	return @proc_list; 
}

sub get_call_statement($proc_name, \@arr_var,$ret_val_str){
	my ($proc_name, $arr_var, $ret_val_str) = (@_[0], @_[1], @_[2] );



	chomp($proc_name); 
	chomp($ret_val_str); 


	my $ret_str = ""; 	# to save the ret string 

	$ret_str .= "### ". $proc_name ." ###". "\n" ; 
	$ret_str .= "ID_P = \"". $proc_name ."\"". "\n"; 



	if ( @$arr_var == 0 ){


#		$ret_str .= 'project_name = rc.callProcedure( ID_P , {} )' . "\n"; 

		$ret_str .=  $ret_val_str . ' = ' . 'rc.callProcedure( ID_P , ' . '{}'  . ' )' ."\n" if $ret_val_str;
		$ret_str .=  $ret_val_str . '' . 'rc.callProcedure( ID_P , ' . '{}'  . ' )' ."\n" if !$ret_val_str;



	}
	else{
		my $ret_params = ""; 
		my $i = 0;
		for ( $i = 0 ; $i<@$arr_var; $i++ ){
			my @var_nm_default_temp = &detach_var_state(@$arr_var[$i]); 
			$ret_params .= "\"" . @var_nm_default_temp[0] . "\"" .':' .  @var_nm_default_temp[0] . ',' ; 
			$ret_str .= @var_nm_default_temp[0] . ' = ' . "\"@var_nm_default_temp[1]\""  . "\n" ; 



		} # end for 

		my $ret_params_str_all = ""; 
		$ret_params_str_all .= '{'; 
		$ret_params_str_all .= $ret_params; 
		$ret_params_str_all .= '}'; 
		$ret_str .=  $ret_val_str . ' = ' . 'rc.callProcedure( ID_P , ' . $ret_params_str_all . ' )' ."\n" if $ret_val_str; 
		$ret_str .=  $ret_val_str . ''    . 'rc.callProcedure( ID_P , ' . $ret_params_str_all . ' )' ."\n" if ! $ret_val_str; 


	} # end else ()

	$ret_str .= "\n"; 

	return $ret_str; 


}


sub from_pattern_to_precise(\@fun_pattern_list, \@proc_list){
	my ($fun_pattern_, $proc_list_) = ( @_[0], @_[1] ); 
	my @fun_pattern = @$fun_pattern_; 
	my @proc_list = @$proc_list_ ; 
	my $i ; my $j; 
	my @arr_fun_ret = (); 
	for( $i = 0 ; $i< @fun_pattern; $i++){
		for( $j=0; $j< @proc_list; $j++ ){
			my $temp_pattern = @fun_pattern[$i]; 
			if ( @proc_list[$j] =~ m/$temp_pattern/ ){
				push @arr_fun_ret , @proc_list[$j]; 
			} # end if ()

		}
	}

	return @arr_fun_ret ; 

}



sub grep_by_each_function_name(\@arr_file, $one_fun_name){
	my ($arr_file_, $one_fun_name ) = ( @_[0], @_[1] ); 
	my @arr_file = @$arr_file_ ; 


	my $proc_name = $one_fun_name ;

	my $proc_name_stat = q(name\=\").$proc_name.q(\"\>); 
	my $proc_name_stat_end = '\<\/Procedure\>'; 
	my $i =0; 

	my $proc_line_found = ''; 
	for ( $i = 0; $i < @arr_file; $i++){
		if (@arr_file[$i] =~ m/$proc_name_stat/ ){
			$proc_line_found = @arr_file[$i]; 
			last; 
		}

	}
	my $proc_line_start = $i;
	my $len = @arr_file ; 
	if ( $proc_line_start == $len || $proc_line_start == $len-1 || $proc_line_start==$len-2){
		die "- no found of $proc_name\n"; 

	}




	for( $i = $proc_line_start; $i< @arr_file; $i++){
		if (@arr_file[$i] =~ m/$proc_name_stat_end/ ){
			last; 
		}

	}

	my $proc_line_end = $i ; 

	for ($i = $proc_line_start-1 ; $i< $proc_line_end+1; $i++){
#print "$i: @arr_file[$i]"; 
	}

	my $var_state_pattern  = q(\<variable name\=\".*\<\/variable\>); 

	my @arr_roi = @arr_file[$proc_line_start-1...$proc_line_end]; 
	my @arr_variables = grep m/$var_state_pattern/, @arr_roi; 

	my $ret_val_pattern =  '\<ReturnStep.*retval\=\".*\"\/\>' ; 

	my @arr_ret = grep m/$ret_val_pattern/, @arr_roi; 
	my $ret_val_statement = @arr_ret[0]; 

	my  $ret_val = &get_ret_val($ret_val_statement);




#print @arr_variables; 


	my $proc_name = &get_procedure_name( $proc_line_found); 
	#my @detached_var = &detach_var_state ( @arr_variables[0] ); 


	print &get_call_statement($proc_name, \@arr_variables,$ret_val); 

}
