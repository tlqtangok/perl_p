# jd add

#!perl
### use list ###
use strict;
use POSIX qw(mkdir assert);
use feature qw(say);
use File::Path qw(make_path remove_tree);

### global var ###
my $HOME = $ENV{HOME};
my $jupyter_nb_port = 10249; 

my $fn_run_history = "$HOME/god_script_for_boot.sh";
`rm $fn_run_history ; touch $fn_run_history`;

my $flag_ub = 0;
$flag_ub =  1 if `cat /etc/issue` =~ m/ubuntu/i;


my $whoami = scalar getpwuid $<;
my $jd = "$HOME/jd";
system( qq{ rm -rf $jd } );

&chk_cmd_rc( qq{mkdir -p $jd} );
my $t = "$jd/t";
&chk_cmd_rc( qq{mkdir -p $t} );
&chk_dir($t);

my $sudo = "";
$sudo = "sudo" if $whoami != "root";

my $perl_p="$jd/perl_p";
&chk_cmd_rc( qq{ cd $jd && git clone https://github.com/tlqtangok/perl_p.git } );
&chk_dir($perl_p);

my $nb = "$t/nb";
&chk_cmd_rc( qq{mkdir -p $nb} );
&chk_dir($nb);

# ngrok files
my $ngrok = "$t/ngrok";
&chk_cmd_rc( qq{ cd $t &&  wget http://jidort.gz01.bdysite.com/ngrok.tar.gz && tar xzf ngrok.tar.gz && rm ngrok.tar.gz  } );
&chk_dir($ngrok);

# nb/lnk , nb/yun
my $lnk = "$nb/lnk";
&chk_cmd_rc( qq{mkdir -p $lnk} );
my $yun = "$nb/yun";
&chk_cmd_rc( qq{mkdir -p $yun} );


# set .bashrc
chdir($HOME);

my $cmd_append_bashrc = qq{
set -e
touch $HOME/.bashrc
touch $HOME/.bash_aliases

cp $HOME/.bashrc $HOME/.bashrc.bak
rm $HOME/.bash_aliases

cat -<<EOF_bashrc >>$HOME/.bashrc

### export list, jd ###
export perl_p=$perl_p
export t=$t
export nb=$nb/lnk/nb
alias s_b='source $perl_p/.bash_aliases'
s_b
### end export ###

EOF_bashrc
};

say( q{$cmd_append_bashrc} );
&chk_cmd_rc($cmd_append_bashrc);

&chk_fn( "$HOME/.bashrc" );

# bash_aliases
say ( qq{ cd $perl_p && base64 -d b64.bash_aliases > .bash_aliases && ln -s $perl_p/.bash_aliases $HOME/.bash_aliases } );
&chk_cmd_rc ( qq{ cd $perl_p && base64 -d b64.bash_aliases > .bash_aliases && ln -s $perl_p/.bash_aliases $HOME/.bash_aliases } );
&chk_fn( "$HOME/.bash_aliases" );

# vimrc
say ( qq{ cd $HOME && cp $perl_p/.vimrc ./ } );
&chk_cmd_rc( qq{ cd $HOME && cp $perl_p/.vimrc ./ } );
&chk_fn( "$HOME/.vimrc" );

# change the net?
        # no need

# change the source.list
my $cmd_edit_sources_list = qq
{
set -e
$sudo touch /etc/apt/sources.list.d/raspi.list

$sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
$sudo chmod 0777  /etc/apt/sources.list

$sudo cp /etc/apt/sources.list.d/raspi.list /etc/apt/sources.list.d/raspi.list.bak
$sudo chmod 0777 /etc/apt/sources.list.d/raspi.list

$sudo cp $perl_p/pi_sources.list /etc/apt/sources.list
$sudo cp $perl_p/pi_raspi.list /etc/apt/sources.list.d/raspi.list

};

if ($flag_ub == 1)
{

        $cmd_edit_sources_list = qq
        {
set -e
$sudo touch /etc/apt/sources.list
$sudo touch /etc/apt/sources.list.d/raspi.list
$sudo rm /etc/apt/sources.list.d/raspi.list

$sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
$sudo chmod 0777  /etc/apt/sources.list

$sudo cp $perl_p/aliyun_ub_16.04_sources.list /etc/apt/sources.list
        };


}  # end if flag_ub

say( q{$cmd_edit_sources_list} );
&chk_cmd_rc( qq{$cmd_edit_sources_list} );


# apt-get update
say( qq{ $sudo apt-get update } );
&chk_cmd_rc( qq{ $sudo apt-get update } );


# install samba
say( qq{$sudo apt install samba} );
&chk_cmd_rc( qq{$sudo apt install samba} );

# edit samba daemon config
my $cmd_set_smb_pw = "$sudo smbpasswd -a $whoami";
say $cmd_set_smb_pw;
&chk_cmd_rc( qq{$cmd_set_smb_pw}  );

my $cmd_edit_samba_config_restart_daemon = qq
{
set -e
$sudo cp /etc/samba/smb.conf /etc/samba/smb.conf.bak
$sudo chmod 0777 /etc/samba/smb.conf
cat -<<EOF_samba >>/etc/samba/smb.conf

[t]
   comment = t
   path = $t
   browseable = yes
   writable = yes
   guest ok = no
   public = no
   force user = $whoami
   force group = $whoami

EOF_samba

$sudo /etc/init.d/smbd restart
};

&chk_fn( "/etc/samba/smb.conf" );
say( q{$cmd_edit_samba_config_restart_daemon} );
&chk_cmd_rc( qq{$cmd_edit_samba_config_restart_daemon} );

# install samba client
say( qq{$sudo apt install smbclient} );
&chk_cmd_rc( qq{$sudo apt install smbclient} );

# install jupyter-notebook
say( qq{$sudo apt install jupyter-notebook} );
&chk_cmd_rc( qq{$sudo apt install jupyter-notebook} );

# config jupyter-notebook 
say( qq{ $sudo jupyter-notebook --generate-config } );
&chk_cmd_rc( qq{ $sudo yes | $sudo jupyter-notebook --generate-config } );
if ($flag_ub)
{
&chk_cmd_rc( qq{ $sudo cp $perl_p/ub_pw_jd_jupyter_notebook_config.py $HOME/.jupyter/jupyter_notebook_config.py } );
}
else
{
&chk_cmd_rc( qq{ $sudo cp $perl_p/pi_pw_pi_jupyter_notebook_config.py $HOME/.jupyter/jupyter_notebook_config.py } );
}

&chk_cmd_rc( qq{ perl  -i.bak -pe 's|(c.NotebookApp.notebook_dir.*u.)(.*)(.)|\\1$nb\\3|; s|(c.NotebookApp.port.*)( \\d.*)|\\1 $jupyter_nb_port|'  $HOME/.jupyter/jupyter_notebook_config.py } );

# install nbconvert
say( qq{$sudo pip install nbconvert} );
&chk_cmd_rc( qq{$sudo pip install nbconvert} );


### sub list ###
sub chk_fn($fn_path)
{
        my $fn_path = shift;
        chomp($fn_path);
        die "- not exist $fn_path\n" if ! -f $fn_path;
}

sub chk_dir($dir_path)
{
        my $dir_path = shift;
        chomp($dir_path);
        die "- not exist $dir_path\n" if ! -e $dir_path;
}

sub chk_cmd_rc($cmd)
{
        my $cmd = shift;
        my $fn_run_history_ =  $fn_run_history;
        assert( -e $fn_run_history_);

        # chomp($cmd);
        &append_content_to_fn($fn_run_history_, $cmd);

        my $rc = system( $cmd );
        if ($rc != 0)
        {
                say $cmd;
        }
        assert( $rc == 0 );
}

sub append_content_to_fn($fn, $content)
{
        my $fn = shift;
        my $content = shift;
        my $cmd_append = qq{

cat -<<EOF_append >>$fn

$content

EOF_append

                        };

        system( qq{$cmd_append} );
}


# jd end


