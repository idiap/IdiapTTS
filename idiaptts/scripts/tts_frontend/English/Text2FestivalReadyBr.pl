#! /usr/bin/perl

#added this in order to use basename command in order to remove the path from the filenames in prompts file
use File::Basename;

my ($inPlainText, $outFestivalText, $outUttFolder) = @ARGV;




open INFILE, $inPlainText or die " Unable to open the file: $inPlainText\n";
open OUTFILE, ">$outFestivalText" or die " Unable to create the file: $outFestivalText\n";
print OUTFILE "(voice_cstr_uk_nina_3hour_hts)\n";
#print OUTFILE "(voice_roger_hts2010)\n";
while ( $tmp = <INFILE> )
{
	chomp($tmp);
	if (( $tmp =~ /^(.+)\t(.+)/ ) )
	{
		$tmpFileName = $1;
# added this because in the prompts file the filenames are with their path
		$tmpFileName = basename($tmpFileName);
		$tmpText = $2;
		$tmpText =~ s/~/ss/g;
		$tmpText =~ s/\\"//g;
		$tmpText =~ s/"//g;
		print OUTFILE "(utt.save (SynthText \"$tmpText\") \"$outUttFolder/$tmpFileName.utt\")\n";
	}
}
close(INFILE);
print OUTFILE "(quit)\n";
close(OUTFILE);
