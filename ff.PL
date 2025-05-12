#!perl
use strict;
use Data::Dumper; 
use feature qw(say);



my $SP_ELINE = $ENV{SP_ELINE}; 
if (length $SP_ELINE == 0)
{
    $SP_ELINE = " ";
}


=pod
fetch field from txt files

# s
# ,
# |

dir /s /A-D |grepw 202| perl ff.PL -1 -2

perl ff.PL txt.txt -1 -2


*** txt.txt ***
1 2 3
4 5 6
7 8 9
-----

run :
	perl ff.PL txt.txt 1 -1
=>
1 3
4 6
7 9
-----
=cut

die "- need 2 args, perl this_file.PL filename.txt \"arr_cols\"\n" if @ARGV == 0; 


&main();

sub main()
{
    my @fc = ();
    my $start_num = 1; 

#if (@ARGV > 0 && $ARGV[0] =~ m/^[-\d]/)
    if ( @ARGV >= 1 && $ARGV[0] =~ m/^[\-\d]/ && ! -f $ARGV[0])
    {
        @fc = <STDIN>; 
        $start_num = 0; 
    }
    elsif (@ARGV >= 2 && -f $ARGV[0])
    {
        my $fn = @ARGV[0];
        open my $FP, "<", $fn; 
        @fc = <$FP>; 
        close $FP; 
        $start_num = 1; 
    }
    else
    {
        die "- not invalid args\n"; 
    }

    my $ENV_arr_cols=join " ", @ARGV[$start_num .. @ARGV-1]; 
    my @arr_col=split / /,$ENV_arr_cols;
    @arr_col = grep m/\w/,@arr_col;


    my @ans_all = (); 


    my $try_line = 10; 
    if (@fc < $try_line)
    {
        $try_line = @fc-1; 
    }

    my @fc_try_line = @fc[0..$try_line-1]; 

    my @a = split m/\s+/, $fc_try_line[0];
    my $expect_fields = @a;
    my $flag_ok_s = 1;
    for (@fc_try_line)
    {

        my @a = split m/\s+/;
        my $expect_fields_ = @a;

        # print "expect_fields: $expect_fields\n"; 
        # print "expect_fields_: $expect_fields_\n"; 
        if ($expect_fields != $expect_fields_ || $expect_fields_ <= 1)
        {
            $flag_ok_s = 0;  
            last;
        }
    }


    my @a = split m/\s*?\,/, $fc_try_line[0];
    my $expect_fields = scalar @a;
    my $flag_ok_comma = 1;
    for (@fc_try_line)
    {
        my @a = split m/\s*?\,/;
        my $expect_fields_ = @a;
        if ($expect_fields != $expect_fields_ || $expect_fields_ <= 1)
        {
            $flag_ok_comma = 0;  
            last;
        }
    }

    my @a = split m/\s*?\|/, $fc_try_line[0];
    my $expect_fields = @a;
    my $flag_ok_pipe = 1;
    for (@fc_try_line)
    {

        my @a = split m/\s*?\|/;
        my $expect_fields_ = @a;
        if ($expect_fields != $expect_fields_ || $expect_fields_ <= 1)
        {
            $flag_ok_pipe = 0;  
            last;
        }
    }


    # print $flag_ok_s,"\n"; 
    # print $flag_ok_comma,"\n"; 
    # print $flag_ok_pipe,"\n"; 


    for ( @fc ) 
    {
        chomp;

        if (length $_ == 0)
        {
            next;
        }

        my @arr_fc_each = ();
        if ($flag_ok_s) 
        {
            @arr_fc_each = split m/\s+/;
        }
        elsif ($flag_ok_comma)
        {
            # TODO
        
        }
        elsif ($flag_ok_comma)
        {
            @arr_fc_each = split m/\s*?\,/;
        }
        elsif ($flag_ok_pipe)
        {
            @arr_fc_each = split m/\s*?\|/;
        }
        else
        {
            if (length $SP_ELINE >= 1)
            {
                @arr_fc_each = split $SP_ELINE;
            }
            else
            {
                die "Warning: use $SP_ELINE from ENV SP_ELINE\n"; 
            }
        }
        # my @arr_fc_each = split ","; 
        my @ans_each=(); 
        push @ans_each , @arr_fc_each[$_] for @arr_col;
        $_= join "$SP_ELINE", @ans_each;    

        #say  ; 

        push @ans_all, $_; 
    }

    print join "\n", @ans_all;
    print "\n";
# #say ; 


#export arr_col="0 1"  && perl -pe '@arr_col=split / /,@ENV{arr_col}; @t = split; @ans=(); push @ans , @t[$_] for @arr_col; $_= "@ans". "\n";  '  txt.txt 

}


