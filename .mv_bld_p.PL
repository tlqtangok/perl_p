#!perl
use feature qw(say);
#$arch_ver='_201403RC1';
	$arch_ver='_201403_SP1_eng4';
$my_hs_p=@ENV{hs_p};
chdir $my_hs_p;
@p_600=`ls |grep -P "ARC6|AS|625"|grep -v 201`;
@p_700=`ls |grep -P "ARC7"|grep -v 201 `;
@p_em=    `ls |grep em  |grep -v 201 `;
@p_hs=`ls|grep HS |grep -v 201 `; 
#@arr_=@p_600;
@arr_=PERL_MATCH_NOT_RENAME_LIST;
map{
chomp;
$new_p=$_.$arch_ver;
`mv -f "$_" "$new_p" `;
say "mv $_ to $new_p ";
select(undef,undef,undef,0.5);
}@arr_;
