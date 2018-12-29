# split -b 1m large_file.txt

# tor 
ls x[a-c]* | perl -e ' print q(export tor="perl $perl_p/tor.PL").qq(\n\n);  map{ chomp; print qq(\$tor $_ &\n); }<>;print qq(wait\necho end\n)'

# fr
perl -e ' print q(export fr="perl $perl_p/fr.PL").qq(\n\nrm x[a-c]*\n);  map{ chomp; print qq(\$fr jd_$_ &\n); }(191..197); print qq(wait\necho end\n)'


