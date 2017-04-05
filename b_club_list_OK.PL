#!perl 
use feature qw(say);
use strict;
#use warnings; 
# --- fun_list()---
my $WIN7=@ENV{win7};

sub get_file_line {
	my $file_=$_[0];
	open(FP,'<',$file_); 
	my @ret_arr = <FP>;
	close(FP); 	
#say @ret_arr; 
	return @ret_arr; 
};
sub get_mail_line{
	#say "get_mail_line";
	my $mail_; 
	$mail_=$_[-1];
	my @mail_list=split ';',$mail_;
	return @mail_list; 
};
sub get_ARGV_list {
#say "get_ARGV_list";
	my @ret_match=@ARGV; 
	if(@ARGV[0] eq ''){
		my @file_match_dat=get_file_line("$WIN7/b_club_match.dat");
		my $_=sprintf("%s","@file_match_dat");
		s/\W+/ /g; 
		@ret_match=split ' ',$_;
	}
	return @ret_match; 
};
sub get_match_list{
#	say "get_match_list"; 
	my ($_match,$_mail)=@_; 
	my @ret_get_match_list=();
	map{
	my $match_ = $_;
	$match_='Laury' if $match_ ~~m/Laurry/i; 
	$match_='Guanhua' if $match_=~m/Yang/i;
	$match_='Rui Zhang' if $match_=~m/Rui/i or $match_=~m/Zhang/i; 
	map{
	my $mail_ =$_;
	
	
	
	push @ret_get_match_list, $_ if m/\b$match_\b/i; 
	}@$_mail;
	}@$_match;
	return @ret_get_match_list; 
};
sub show_match_list{
	my $num_p = @_;

	$num_p = @ARGV if @ARGV[0] ne "";

	my $today=`date`;
	my $i=1;
	my $total_exp = 	40.00	; 
	my $aver = $total_exp/$num_p;

	say " ";

	say " Total Expense:	$total_exp\n Players Num:	$num_p"; 
	printf(" Average Cost(CNY):	%.2f\n",$aver);
	say "\n------";
	say "!!!match failed !!!\n" if( @_ != $num_p); 
	map{
	say "   $_;";$i++; 
	}@_;
	$i-=1;
	say "------";

	say "\n\n\n$today"; 

	open(OUT,'>',"$WIN7/b_club_match_result.txt");
	say OUT " ";
	say OUT  " Total Expense:	$total_exp\n Players Num:	$num_p\n ";
	say OUT "\n------";
	map{
	say OUT "   $_";$i++; 
	}@_;
	$i-=1;
	say OUT "match failed !!!\n" if( $i != $num_p); 
	say OUT "------";

	say OUT "\n\n\n$today"; 

	close(OUT);
};
# --- main() --- 
#
my $perl_p=@ENV{perl_p};
my @file_line;my @ARGV_list;my @match_list; 
my @mail_line; 
@file_line=get_file_line("$perl_p/b_club_list_OK.PL");
@mail_line=get_mail_line(@file_line);
@ARGV_list=get_ARGV_list(@ARGV);
@match_list=get_match_list(\@ARGV_list,\@mail_line); 
show_match_list(@match_list);
# --- END main ---

#all the name list
##Bing Xu <xubing@synopsys.com>; Laury Li <zhl@synopsys.com>; Qilin Tan <qltan@synopsys.com>; Allon Li <longli@synopsys.com>; Xin Li <lxin@synopsys.com>; Ling Han <linghan@synopsys.com>; Jianying Peng <jypeng@synopsys.com>; Da Fang <dafang@synopsys.com>; Sean Mao <ckmao@synopsys.com>; Darcy Shen <jyshen@synopsys.com>; Bob Hu <zbhu@synopsys.com>; Eric Rao <jlrao@synopsys.com>; Austin Qiu <mqiu@synopsys.com>; Snowy Deng <cndeng@synopsys.com>; Fenglin Guan <flguan@synopsys.com>; Dean Ke <xxke@synopsys.com>; Newton Chen <pengc@synopsys.com>; Richard Zhou <zhoux@synopsys.com>; Mark Cheng <sbcheng@synopsys.com>; Jane Wu <dwu@synopsys.com>; Haifeng Han <haifeng@synopsys.com>; Frank Fan <ykfan@synopsys.com>; Gabriel Tu <sjtu@synopsys.com>; Dan Zhu <danzhu@synopsys.com>; Hongwei Ma <mahw@synopsys.com>; Rui Chen <ruich@synopsys.com>; Alanbrooke Yu <qyyu@synopsys.com>; Guanhua Yang <ygh@synopsys.com>; Rui Zhang <ruizhang@synopsys.com>; Zhe Wang <wangzhe@synopsys.com>; Kevin Chen <chenxh@synopsys.com>; Yang Liu <yangl@synopsys.com>; Tony Zhou <tzhou@synopsys.com>; Yuting Li <liyt@synopsys.com>; Yuekui Bao <ykbao@synopsys.com>;Liyuan Chen <liyuan@synopsys.com>;Binbin Duan <bbduan@synopsys.com>; Jidor Tang <linqi@synopsys.com>; 
