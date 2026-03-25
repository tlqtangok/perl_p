#!/usr/bin/env perl
use strict;
use warnings;
use IO::Socket::INET;

my $PORT = $ARGV[0] || 10270;

my $s = IO::Socket::INET->new(
    LocalPort => $PORT,
    Reuse => 1,
    Listen => 10
) or die "Cannot create socket: $!";

print "Server running at http://localhost:$PORT/\n";

while(my $c = $s->accept) {
    my $r;
    $c->recv($r, 1024);
    my $h = "HTTP/1.0 200 OK\r\nContent-Type: text/plain\r\n\r\n";
    $c->send($h . "Hello World!\n" . localtime() . "\n");
    close($c);
}
