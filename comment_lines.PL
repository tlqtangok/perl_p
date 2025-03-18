#!perl
use File::Spec;
use strict;
use Encode;




my $SP = File::Spec->catfile('', ''); 
$| = 1;


#print join ",", @ARGV;
#print "\n";

&main();
sub main()
{

    # 设置标准输入和标准输出为 UTF-8 编码
    binmode(STDIN, ":encoding(UTF-8)");
    binmode(STDOUT, ":encoding(UTF-8)");

    my @fc = (
        '    ',
        '               ',
        '    printf("- reg_idx: 0x%x, val : 0x%08x\n", reg_idx, reg_val);',
        '               ',
        '    ',
        '               // get file size , by c++',
        '    #if 1',
        '              fstream if_("txt.txt"); assert(if_.is_open());',
        '    ',
        '              f_.seekg(0, ios::end);',
        '       cout << if_.tellg() << endl;;'    ,
        '               ',
        '    if_.close();',
        '  #endif',
        '    ',
        '               ',
        '    // type cast',
        '               ',
        '                volatile uint8_t *pe_result_from_pcie_raw = reinterpret_cast<volatile uint8_t *>(fpga_output_addr_reads.get_buffer());',
        '               ',
        '    ',
        '               ', 

    );


    # perl this_script.PL file.txt "//" 
    # dir | perl this_script.PL "//"
    my $comment_str = "//"; # "#"

    #print @ARGV + 0, "_____\n";

    if (@ARGV>1)
    {
        my $fn = $ARGV[0]; 
        open my $FN,"<",$fn or die;
        @fc = <$FN>;
        close $FN;
        $comment_str = $ARGV[1];
    }
    else
    {
        @fc = <STDIN>;
        $comment_str = $ARGV[0];
    } 

    map{
        chomp;

        s/^\x{FEFF}//;  # strip bom <feff>

        s/\t/    /g; # don't need <TAB>
    }@fc;


    my @fc_copy = (@fc);

    #print "\$comment_str:${comment_str}___\n"; 

    my $ans_str = "NULL_STR"; 

    #$ans_str = join "\n", @fc;
    #print $ans_str, "\n";

    @fc = &switch_comment_for_c_stat(\@fc, $comment_str);
    $ans_str = join "\n", @fc;
    print $ans_str, "\n";



    if (0)
    {
        @fc = &switch_comment_for_c_stat(\@fc, $comment_str);
        $ans_str = join "\n", @fc;
        print $ans_str, "\n";
        for(my $i=0;$i<@fc;$i++)
        {
            if ($fc[$i] ne $fc_copy[$i])
            {
                ${fc[$i]} =~ s/ /0/g;
                ${fc_copy[$i]} =~ s/ /0/g;

                print  "$i is not match\n";
                print "__ ${fc[$i]} __ ne __ ${fc_copy[$i]} __","\n";
            }
        }
    }

}

sub is_utf8($)
{
    my ($data) = @_;
    return !($data =~ /[^\x{0000}-\x{007F}\x{00A0}-\x{D7FF}\x{E000}-\x{FFFF}\x{10000}-\x{10FFFF}]/g);
}

sub is_commented(\@fc,$comment_str)
{
    my $ret_str = "IS_COMMENTED"; 
    #my $ret_str = "NOT_COMMENTED"; 
    my ($fc_ref,$comment_str) = @_;
    my @fc = (@$fc_ref);

    for (@fc)
    {
        if (m/\S/)
        {
            my $loc_comment = index($_, $comment_str." ");
            if ($loc_comment!=-1)
            {
               my $prefix = substr $_, 0, $loc_comment; 
               if ($prefix =~ m/\S/)
               {
                   $ret_str = "NOT_COMMENTED";
               }
            }
            else
            {
                $ret_str = "NOT_COMMENTED";
                last;
            }
        }
    };

    my $ret_code = 0;
    if ($ret_str eq "IS_COMMENTED")
    {
        $ret_code = 1;
    }
    else
    {
        $ret_code = 0;
    }
    return  $ret_code;
}

sub switch_comment_for_c_stat(\@fc, $comment_str)
{
    my ($fc_ref, $comment_str) = @_;

    my @fc = (@$fc_ref);



    my @fc_copy = (@fc); 

    my $is_comment = &is_commented(\@fc, $comment_str);
    #print "\$is_comment: $is_comment\n"; 

    if ($is_comment)
    {
        # uncomment process
        my %h_loc = ();

        my $max_cnt_loc = 0;
        my $max_loc = -1;

        #print "\$comment_str:${comment_str}___\n"; 

        
        map{


            my $index_c = index($_, $comment_str . " ");

            if ($index_c != -1)
            {
                $h_loc{$index_c}++;
                if ($max_cnt_loc < $h_loc{$index_c})
                {
                    $max_cnt_loc = $h_loc{$index_c};
                    $max_loc = $index_c;
                }
            }

        }@fc;

        #print "\$max_loc: $max_loc\n";

        map{

            #my $old_s = $_;
            #$old_s =~ s/\s/0/g;
            #print $old_s , "\n";

            my $index_c = index($_, $comment_str . " ");

            if ($index_c != -1)
            {
                my @a = split m//;

                $a[$max_loc+0] = "";
                $a[$max_loc+1] = "";

                if ($comment_str eq "//")
                {
                    $a[$max_loc+2] = "";
                }

                $_ = join "",@a;
            }

        }@fc;
    }
    else
    {
        # comment process
        my $space_len_min = 100;

        map{
            chomp;  
            my $old_str = $_;
            if (m/\S/)
            {
                s/^(\s*?)\S.*$/\1/;
                my $len_t = scalar length $_;
                #print "old_str: __${old_str}__, len_t: $len_t\n"; 

                if ($space_len_min >=  $len_t && $old_str =~ m/\S/)
                {
                    $space_len_min = $len_t;
                }
            }
        }@fc_copy;

        #print "\n\n";


        #print "\$space_len_min: $space_len_min\n";

        map{
            my $old_str = $_;
                if (m/\S/)
                {
                    #print "\$old_str:$old_str, \$space_len_min: $space_len_min\n";
                    my @a=split m//,$old_str;

                    for(my $i=0;$i<@a;$i++)
                    {
                        #print "i:$i, v:$a[$i];";
                    }
                    #print "\n";

                    $a[$space_len_min] = $comment_str ." ". $a[$space_len_min]; 

                    #print "new:\n";

                    for(my $i=0;$i<@a;$i++)
                    {
                        #print "i:$i, v:$a[$i];";
                    }
                    #print "\n";
                    $_ = join "",@a;

                    #print join ",", @a;
                    #print "\n";
                }
        }@fc;

    }


    return @fc;

}
sub get_stderr(@)
{
my @fc = @_;
return grep{m/run \[.*error/;}@fc;
}
sub cml_to_file($cml)
{
    my $cml = shift; 
    my $fn_cml = $ENV{tmp}."$SP"."run_e_cml.bat";
    $| = 1;

    chomp($cml);

    my @fc_cml = split m/\n/, $cml;

    map {
        my $org = $_;
        chomp;
        s/^\s*?(\w)/\1/g;
        s/\s+$//g;

        if (length($_)>0)
        {
            my $add_if_stat = qq{if %errorlevel% NEQ 0 (echo run [ $_ ] error ! && exit /b 1)}. "\n"; 
            my $cml_all = $_ . "\n". $add_if_stat;
            $_ = $cml_all;
        }
        else
        {
            $_ = "\n"; 
        }

    }@fc_cml;

    open my $FILE_cml , ">" , "$fn_cml" or die "open $fn_cml\n";
    print $FILE_cml '@echo off'."\n";
    print $FILE_cml join "",@fc_cml;
    print $FILE_cml '@echo on'."\n";
    close $FILE_cml;

    return $fn_cml; 
}

sub cml_tol($cml)
{
    my $cml = shift;
    chomp($cml);
    chomp($cml);
    $cml =~ s/\n\s*?$//g;

    $cml =~ s/\n/&& /g;
    return $cml;
}
sub run_cmd($cml)
{
    my $cml = shift; 

    my @fc  = ` $cml  2>&1 `; 
    my $ret_code = $? >> 8; 
    my @stderr = grep {m/run \[.*error/;}@fc;
    #print "@stderr","\n";
    return ($ret_code, @fc); 
}

sub say($)
{
    my $str = ""; 
    if (@_ > 0)
    {
        $str = join "", @_;
    }
    print $str, "\n";
}


