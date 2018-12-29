# split -b 1m large_file.txt

# tor 
ls xa* | perl -e ' print q(export fr="perl $perl_p/tor.PL").qq(\n\n);  map{ chomp; print qq(\$tor $_ &\n); }<>;'

# fr
perl -e ' print q(export fr="perl $perl_p/fr.PL").qq(\n\n);  map{ chomp; print qq(\$fr jd_$_ &\n); }(191..197);'


