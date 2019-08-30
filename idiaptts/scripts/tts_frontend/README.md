Scripts to make HTS compatible labels from text files.

* For English, usage is:  
Unpack festival_files.tar.gz in your festival directory.
Download the unilex dictionary from [http://www.cstr.ed.ac.uk/projects/unisyn/](http://www.cstr.ed.ac.uk/projects/unisyn/).
Copy the files from *festival/lib/dicts/unilex/* into your festival directory at the same location *full_festival_location/lib/dicts/unilex/*.
The program requires the unilex dictionary for English and comes with two speakers uk_nina (British accent) and clb (American accent). To run the example (accent can be AM or BR):  
$ English/makeLabels.sh festival_dir English/example_English_prompts.txt accent ~/test_labels

