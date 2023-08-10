#!perl
use feature qw(say);
use POSIX qw(assert);
 

&main(); 


### sub list ###
sub main()
{ 
    say "- in main()"; 

    my $hpp_cpp_from = q{D:\jd\t\vs\tma\tma\tma.cpp}; 

    my $hpp_to = q{D:\jd\t\vs\tma\tma\PCA_0719\Algoritm.h};
    my $cpp_to = q{D:\jd\t\vs\tma\tma\PCA_0719\Algoritm.cpp};



    my $label_com_lib_hpp = "_com_lib_ hpp"; 
    my $label_com_lib_cpp = "_com_lib_ cpp"; 

    my $label_tma_hpp = "tma_ hpp"; 
    my $label_tma_cpp = "tma_ cpp"; 

    my @fc_from = &read_file_to_fc($hpp_cpp_from); 
    my @fc_hpp_to = &read_file_to_fc($hpp_to); 
    my @fc_cpp_to = &read_file_to_fc($cpp_to); 


    my $flag_ret_full = 0; 

    my (@b_from_com_lib_hpp, @se_, @b_from_com_lib_cpp, @b_from_tma_hpp, @b_from_tma_cpp); 

    (@b_from_com_lib_hpp) = &find_label_block($label_com_lib_hpp, $flag_ret_full, \@fc_from); 
    (@b_from_com_lib_cpp) = &find_label_block($label_com_lib_cpp, $flag_ret_full, \@fc_from); 

    (@b_from_tma_hpp) = &find_label_block($label_tma_hpp, $flag_ret_full, \@fc_from); 
    (@b_from_tma_cpp) = &find_label_block($label_tma_cpp, $flag_ret_full, \@fc_from); 


    $flag_ret_full++;

    (@se_) = &find_label_block($label_com_lib_hpp, $flag_ret_full,  \@fc_hpp_to); 
    @fc_hpp_to = (@fc_hpp_to[0 ... ($se_[0]-1)],  (@b_from_com_lib_hpp), @fc_hpp_to[($se_[1]+1) ... (@fc_hpp_to-1)]);
    (@se_) = &find_label_block($label_tma_hpp, $flag_ret_full,  \@fc_hpp_to); 
    @fc_hpp_to = (@fc_hpp_to[0 ... ($se_[0]-1)],  (@b_from_tma_hpp), @fc_hpp_to[($se_[1]+1) ... (@fc_hpp_to-1)]);


    (@se_) = &find_label_block($label_com_lib_cpp, $flag_ret_full, \@fc_cpp_to); 
    @fc_cpp_to = (@fc_cpp_to[0 ... ($se_[0]-1)],  (@b_from_com_lib_cpp), @fc_cpp_to[($se_[1]+1) ... (@fc_cpp_to-1)]);
    (@se_) = &find_label_block($label_tma_cpp, $flag_ret_full,  \@fc_cpp_to); 
    @fc_cpp_to = (@fc_cpp_to[0 ... ($se_[0]-1)],  (@b_from_tma_cpp), @fc_cpp_to[($se_[1]+1) ... (@fc_cpp_to-1)]);

    &print_to_file(\@fc_hpp_to, $hpp_to); 
    &print_to_file(\@fc_cpp_to, $cpp_to); 
}



sub  print_to_file(\@fc, $fn)
{
    my $r_fc = shift;
    my $fn = shift;
    open my $FP, ">", $fn or die "- cannot open $fn"; 
    print $FP @$r_fc;
    close $FP; 
    say "- print updated content to $fn"; 
}

sub find_label_block($label, $flag, $@fc)
{
    my $label = shift; 
    my $flag_ret_full = shift;
    my $r_fc = shift; 

    my @si_ei = (0,0); 

    my $cnt_si_ei = 0; 
    my $cnt_fc = 0; 

    for (@$r_fc)
    {
        if ($_ =~ m/$label/)
        {
            $si_ei[$cnt_si_ei] = $cnt_fc;  
            $cnt_si_ei++; 
        }
        $cnt_fc++; 
    }


    
    my @ret = (@$r_fc[$si_ei[0]..$si_ei[1]]); 

    if ($flag_ret_full == 1)
    {
        @ret = (@si_ei); 
    }

    return @ret; 
}


sub read_file_to_fc($)
{
    my $fn = shift; 
    open my $FP_from , "<", $fn or die; 
    my @fc_fp_from = <$FP_from>; 
    close $FP_from;
    return @fc_fp_from; 
}


