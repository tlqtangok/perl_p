#!/usr/bin/env perl
use strict;
use warnings;
use IO::Socket::INET;
use IO::Select;
use POSIX qw(strftime);

$| = 1;  # Autoflush

my $PORT = $ARGV[0] || 10270;

sub get_timestamp {
    return strftime("%Y%m%d_%H%M", localtime);
}

sub log_msg {
    my ($msg) = @_;
    print "[" . get_timestamp() . "] $msg\n";
}

my $s = IO::Socket::INET->new(
    LocalPort => $PORT,
    Listen => 10,
    ReuseAddr => 1
) or die "Cannot create socket: $!";

log_msg("Server running at http://localhost:$PORT/");

while(my $c = $s->accept) {
    my $client_addr = $c->peerhost;
    my $client_port = $c->peerport;
    log_msg("SERVER: Connection from $client_addr:$client_port");
    
    my $sel = IO::Select->new($c);
    my $request = '';
    
    while($sel->can_read(5)) {
        my $buf;
        my $n = sysread($c, $buf, 8192);
        last unless defined $n && $n > 0;
        $request .= $buf;
        last if $request =~ /\r\n\r\n/s;
    }
    
    if ($request =~ /^(\S+)\s+(\S+)/) {
        log_msg("SERVER: $1 $2");
    }
    
    my $body = "[" . get_timestamp() . "] hello world\n";
    print $c "HTTP/1.1 200 OK\r\n";
    print $c "Content-Type: text/plain\r\n";
    print $c "Content-Length: " . length($body) . "\r\n";
    print $c "\r\n";
    print $c $body;
    
    log_msg("SERVER: Sent " . length($body) . " bytes");
    
    close($c);
}
