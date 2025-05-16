#!perl

use utf8;
use Encode qw(decode encode);

# Entry point - call the main function with command line arguments
main(@ARGV);

# Simple encoding detection without external modules
sub detect_encoding
{
    my $file = shift;
    my $encoding = 'cp1252'; # Default Windows encoding
    
    # Try to detect encoding
    open my $fh, '<:raw', $file or return $encoding;
    binmode($fh);
    
    # Read the first chunk to detect encoding
    my $chunk;
    read($fh, $chunk, 4096);
    close $fh;
    
    # Check for UTF-8 BOM
    if (substr($chunk, 0, 3) eq "\xEF\xBB\xBF")
    {
        return 'utf8-BOM';
    }
    
    # Simple heuristic for UTF-8 detection
    if (is_valid_utf8($chunk))
    {
        return 'utf8';
    }
    
    # For Chinese text, check for common encodings
    if ($^O eq "MSWin32")
    {
        return 'cp936'; # Common for Chinese Windows
    }
    
    return $encoding;
}

# Simple UTF-8 validation function
sub is_valid_utf8
{
    my $string = shift;
    
    # Count of multi-byte sequences
    my $utf8_count = 0;
    my $ascii_count = 0;
    
    for (my $i = 0; $i < length($string); $i++)
    {
        my $byte = ord(substr($string, $i, 1));
        
        # ASCII range
        if ($byte <= 0x7F)
        {
            $ascii_count++;
            next;
        }
        
        # Start of 2-byte sequence (110xxxxx)
        if (($byte & 0xE0) == 0xC0)
        {
            $i++;
            if ($i >= length($string)) { return 0; }
            my $byte2 = ord(substr($string, $i, 1));
            # Second byte must be 10xxxxxx
            if (($byte2 & 0xC0) != 0x80) { return 0; }
            $utf8_count++;
            next;
        }
        
        # Start of 3-byte sequence (1110xxxx)
        if (($byte & 0xF0) == 0xE0)
        {
            $i += 2;
            if ($i >= length($string)) { return 0; }
            my $byte2 = ord(substr($string, $i-1, 1));
            my $byte3 = ord(substr($string, $i, 1));
            # Second and third bytes must be 10xxxxxx
            if (($byte2 & 0xC0) != 0x80 || ($byte3 & 0xC0) != 0x80) { return 0; }
            $utf8_count++;
            next;
        }
        
        # Start of 4-byte sequence (11110xxx)
        if (($byte & 0xF8) == 0xF0)
        {
            $i += 3;
            if ($i >= length($string)) { return 0; }
            my $byte2 = ord(substr($string, $i-2, 1));
            my $byte3 = ord(substr($string, $i-1, 1));
            my $byte4 = ord(substr($string, $i, 1));
            # Second, third and fourth bytes must be 10xxxxxx
            if (($byte2 & 0xC0) != 0x80 || 
                ($byte3 & 0xC0) != 0x80 || 
                ($byte4 & 0xC0) != 0x80) { return 0; }
            $utf8_count++;
            next;
        }
        
        # Invalid UTF-8 pattern
        return 0;
    }
    
    # If we found more than a few UTF-8 sequences and no invalid patterns
    return ($utf8_count > 3);
}

# Set up output encoding for STDOUT
sub setup_output_encoding
{
    my $use_utf8 = shift;
    
    if (!-t STDOUT)
    {
        if ($use_utf8)
        {
            # Use UTF-8 with BOM for output
            binmode(STDOUT, ":utf8");
            if ($^O eq "MSWin32")
            {
                # UTF-8 BOM
                print "\x{FEFF}"; 
            }
        }
        else
        {
            # Use system default encoding
            if ($^O eq "MSWin32")
            {
                # Windows typically uses cp1252 or similar
                binmode(STDOUT, ":encoding(cp1252)");
            }
            else
            {
                # Use locale setting on Unix-like systems
                binmode(STDOUT, ":encoding(locale)");
            }
        }
    }
}

# Process a single file with appropriate encoding
sub process_file
{
    my ($file, $encoding, $use_utf8_output) = @_;
    
    print "*********************\n";
    print "$file\n";
    print "*********************\n";
    
    if ($^O eq "MSWin32")
    {
        # Use Perl's file handling with detected encoding
        my $enc_layer = ($encoding eq 'utf8-BOM') ? '<:utf8' : "<:encoding($encoding)";
        if (open my $fh, $enc_layer, $file)
        {
            # Skip BOM if present
            if ($encoding eq 'utf8-BOM')
            {
                my $first_char = getc($fh);
                if (ord($first_char) != 0xFEFF)
                {
                    # If not BOM
                    seek($fh, 0, 0); # Reset to beginning
                }
            }
            
            # Read and print file contents
            while (<$fh>)
            {
                print $_;
            }
            close $fh;
        }
        else
        {
            # Fallback to system command
            print `type "$file" 2>nul`;
        }
    }
    else
    {
        # For non-Windows, use cat but handle encoding
        if ($use_utf8_output)
        {
            # Convert to UTF-8 if needed
            if (open my $fh, "<:encoding($encoding)", $file)
            {
                while (<$fh>)
                {
                    print $_;
                }
                close $fh;
            }
            else
            {
                print `cat "$file"`;
            }
        }
        else
        {
            # Use cat for original encoding
            print `cat "$file"`;
        }
    }
    
    print "\n\n";
}

# Main function to process files
sub main
{
    my @files = @_;
    
    # Skip if no files to process
    return unless @files;
    
    # First pass - check for UTF-8 files
    my $found_utf8 = 0;
    foreach my $file (@files)
    {
        next unless -T $file;
        
        my $encoding = detect_encoding($file);
        if ($encoding eq 'utf8' || $encoding eq 'utf8-BOM')
        {
            $found_utf8 = 1;
            last; # One UTF-8 file is enough to trigger UTF-8 output
        }
    }
    
    # Set up output encoding
    setup_output_encoding($found_utf8);
    
    # Process each file
    foreach my $file (@files)
    {
        if (-T $file)
        {
            my $encoding = detect_encoding($file);
            process_file($file, $encoding, $found_utf8);
        }
    }
    
    print "EOF\n\n";
}

