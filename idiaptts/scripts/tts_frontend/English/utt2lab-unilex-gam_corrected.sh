#!/bin/tcsh -f
#  ----------------------------------------------------------------  #
#      The HMM-Based Speech Synthesis System (HTS): version 1.1.1    #
#                        HTS Working Group                           #
#                                                                    #
#                   Department of Computer Science                   #
#                   Nagoya Institute of Technology                   #
#                                and                                 #
#    Interdisciplinary Graduate School of Science and Engineering    #
#                   Tokyo Institute of Technology                    #
#                      Copyright (c) 2001-2003                       #
#                        All Rights Reserved.                        #
#                                                                    #
#  Permission is hereby granted, free of charge, to use and          #
#  distribute this software and its documentation without            #
#  restriction, including without limitation the rights to use,      #
#  copy, modify, merge, publish, distribute, sublicense, and/or      #
#  sell copies of this work, and to permit persons to whom this      #
#  work is furnished to do so, subject to the following conditions:  #
#                                                                    #
#    1. The code must retain the above copyright notice, this list   #
#       of conditions and the following disclaimer.                  #
#                                                                    #
#    2. Any modifications must be clearly marked as such.            #
#                                                                    #
#  NAGOYA INSTITUTE OF TECHNOLOGY, TOKYO INSITITUTE OF TECHNOLOGY,   #
#  HTS WORKING GROUP, AND THE CONTRIBUTORS TO THIS WORK DISCLAIM     #
#  ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL        #
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT    #
#  SHALL NAGOYA INSTITUTE OF TECHNOLOGY, TOKYO INSITITUTE OF         #
#  TECHNOLOGY, HTS WORKING GROUP, NOR THE CONTRIBUTORS BE LIABLE     #
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY         #
#  DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,   #
#  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTUOUS    #
#  ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR           #
#  PERFORMANCE OF THIS SOFTWARE.                                     #
#                                                                    #
#  ----------------------------------------------------------------  #
#    utt2lab.sh  : convert festival utt file into context-dependent  #
#                  label and context-independent segment label file  #
#                  for HMM-based speech synthesis                    #
#                  # US version                                      #
#                                    2003/12/26 by Heiga Zen         #
#  ----------------------------------------------------------------  #
#           Add functions supporting for                             #
#                 - Unilex                                           #
#                 - Pause                                            #
#                 - Vowel reduction and reduced form                 #
#                                    July 2008 Junichi Yamagishi     #
# -----------------------------------------------------------------  #


set speaker   = $argv[1]	
set basedir   = .
set estdir    = $argv[2] 
set dumpfeats = $estdir/examples/dumpfeats
set scpdir    = . 

# Work in temporary directory.
set -r tmp=`mktemp -d#`
onintr int  # Go to label 'int:' when Ctrl-C.

set src  = $basedir/utt
set mono = $basedir/labels/mono
set full = $basedir/labels/full

mkdir -p $basedir/labels
mkdir -p $mono
mkdir -p $full

cat <<EOF >! $tmp/label.feats
;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; SEGMENT

;; {p, c, n}.name
    p.name                                                  ;  1 
    name                                                    ;  2 
    n.name                                                  ;  3

;; position in syllable (segment)
    pos_in_syl                                              ;  4

;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; SYLLABLE

;; {p, c, n}.stress
    R:SylStructure.parent.R:Syllable.p.stress               ;  5
    R:SylStructure.parent.R:Syllable.stress                 ;  6
    R:SylStructure.parent.R:Syllable.n.stress               ;  7

;; {p, c, n}.accent
    R:SylStructure.parent.R:Syllable.p.accented             ; 8
    R:SylStructure.parent.R:Syllable.accented               ; 9
    R:SylStructure.parent.R:Syllable.n.accented             ; 10

;; {p, c, n}.length (segment)
    R:SylStructure.parent.R:Syllable.p.syl_numphones        ; 11
    R:SylStructure.parent.R:Syllable.syl_numphones          ; 12
    R:SylStructure.parent.R:Syllable.n.syl_numphones        ; 13

;; position in word (syllable)
    R:SylStructure.parent.R:Syllable.pos_in_word            ; 14

;; position in phrase (syllable)
    R:SylStructure.parent.R:Syllable.syl_in                 ; 15
    R:SylStructure.parent.R:Syllable.syl_out                ; 16

;; position in phrase (stressed syllable)
    R:SylStructure.parent.R:Syllable.ssyl_in                ; 17
    R:SylStructure.parent.R:Syllable.ssyl_out               ; 18

;; position in phrase (accented syllable)
    R:SylStructure.parent.R:Syllable.asyl_in                ; 19
    R:SylStructure.parent.R:Syllable.asyl_out               ; 20

;; distance from stressed syllable                 
    R:SylStructure.parent.R:Syllable.lisp_distance_to_p_stress   ; 21
    R:SylStructure.parent.R:Syllable.lisp_distance_to_n_stress   ; 22

;; distance to accented syllable (syllable)                 
    R:SylStructure.parent.R:Syllable.lisp_distance_to_p_accent   ; 23
    R:SylStructure.parent.R:Syllable.lisp_distance_to_n_accent   ; 24  

;; name of the vowel of syllable
    R:SylStructure.parent.R:Syllable.lisp_syl_reduced_vowel      ; 25

;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; WORD

;; {p, c, n}.gpos
    R:SylStructure.parent.parent.R:Word.p.gpos              ; 26
    R:SylStructure.parent.parent.R:Word.gpos                ; 27
    R:SylStructure.parent.parent.R:Word.n.gpos              ; 28

;; {p, c, n}.length (syllable)
    R:SylStructure.parent.parent.R:Word.p.word_numsyls      ; 29
    R:SylStructure.parent.parent.R:Word.word_numsyls        ; 30
    R:SylStructure.parent.parent.R:Word.n.word_numsyls      ; 31

;; position in phrase (word)
    R:SylStructure.parent.parent.R:Word.pos_in_phrase       ; 32
    R:SylStructure.parent.parent.R:Word.words_out           ; 33

;; position in phrase (content word)
    R:SylStructure.parent.parent.R:Word.content_words_in    ; 34
    R:SylStructure.parent.parent.R:Word.content_words_out   ; 35

;; distance to content word (word)
    R:SylStructure.parent.parent.R:Word.lisp_distance_to_p_content ; 36
    R:SylStructure.parent.parent.R:Word.lisp_distance_to_n_content ; 37

;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; PHRASE

;; {p, c, n}.length (syllable)
    R:SylStructure.parent.parent.R:Phrase.parent.p.lisp_num_syls_in_phrase ; 38
    R:SylStructure.parent.parent.R:Phrase.parent.lisp_num_syls_in_phrase   ; 39
    R:SylStructure.parent.parent.R:Phrase.parent.n.lisp_num_syls_in_phrase ; 40

;; {p, c, n}.length (word)
    R:SylStructure.parent.parent.R:Phrase.parent.p.lisp_num_words_in_phrase; 41
    R:SylStructure.parent.parent.R:Phrase.parent.lisp_num_words_in_phrase  ; 42
    R:SylStructure.parent.parent.R:Phrase.parent.n.lisp_num_words_in_phrase; 43

;; position in major phrase (phrase)
    R:SylStructure.parent.R:Syllable.sub_phrases            ; 44

;; type of end tone of this phrase
    R:SylStructure.parent.parent.R:Phrase.parent.daughtern.R:SylStructure.daughtern.tobi_endtone ; 45

;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; UTTERANCE

;; length (syllable)
    lisp_total_syls                                         ; 46

;; length (word)
    lisp_total_words                                        ; 47

;; length (phrase)
    lisp_total_phrases                                      ; 48

;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; for "#"

    p.R:SylStructure.parent.R:Syllable.stress               ; 49
    n.R:SylStructure.parent.R:Syllable.stress               ; 50

    p.R:SylStructure.parent.R:Syllable.accented             ; 51
    n.R:SylStructure.parent.R:Syllable.accented             ; 52

    p.R:SylStructure.parent.R:Syllable.syl_numphones        ; 53
    n.R:SylStructure.parent.R:Syllable.syl_numphones        ; 54

    p.R:SylStructure.parent.parent.R:Word.gpos              ; 55
    n.R:SylStructure.parent.parent.R:Word.gpos              ; 56

    p.R:SylStructure.parent.parent.R:Word.word_numsyls      ; 57
    n.R:SylStructure.parent.parent.R:Word.word_numsyls      ; 58

    p.R:SylStructure.parent.parent.R:Phrase.parent.lisp_num_syls_in_phrase ; 59
    n.R:SylStructure.parent.parent.R:Phrase.parent.lisp_num_syls_in_phrase ; 60

    p.R:SylStructure.parent.parent.R:Phrase.parent.lisp_num_words_in_phrase ; 61
    n.R:SylStructure.parent.parent.R:Phrase.parent.lisp_num_words_in_phrase ; 62

;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; additional feature

;; for quinphone

    p.p.name                                                ; 63
    n.n.name                                                ; 64

;; boundary

    segment_start                                           ; 65
    segment_end                                             ; 66

;; pbreak information ( pp p c n nn ) 

    p.p.p.R:SylStructure.parent.parent.R:Word.pbreak        ; 67 (for pp)
    p.p.R:SylStructure.parent.parent.R:Word.pbreak          ; 68 (for p)
    p.R:SylStructure.parent.parent.R:Word.pbreak            ; 69 (for c)
    R:SylStructure.parent.parent.R:Word.pbreak              ; 70 (for n)
    n.R:SylStructure.parent.parent.R:Word.pbreak            ; 71 (for nn)

;; {p, c, n}.length (syllable)

    p.p.p.R:SylStructure.parent.parent.R:Word.word_numsyls    ; 72
    p.p.R:SylStructure.parent.parent.R:Word.word_numsyls      ; 73 
    p.R:SylStructure.parent.parent.R:Word.word_numsyls        ; 74 
    R:SylStructure.parent.parent.R:Word.word_numsyls          ; 75 
    n.R:SylStructure.parent.parent.R:Word.word_numsyls        ; 76 
    n.n.R:SylStructure.parent.parent.R:Word.word_numsyls      ; 77 
    n.n.n.R:SylStructure.parent.parent.R:Word.word_numsyls    ; 78

;; Reducable
    p.p.reducable                                             ; 79 
    p.reducable                                               ; 80 
    reducable                                                 ; 81
    n.reducable                                               ; 82
    n.n.reducable                                             ; 83

;; Reduced form 
    p.p.reducedform                                           ; 84
    p.reducedform                                             ; 85 
    reducedform                                               ; 86
    n.reducedform                                             ; 87
    n.n.reducedform                                           ; 88

EOF
 
cat <<EOF >! $tmp/label-full.awk
{
##############################
###  SEGMENT

#  boundary
   printf "%10.0f %10.0f ", 1e7 * \$65, 1e7 * \$66

#  pp.name
    printf "%s",  \$63 == "0" ? "xx" : (\$63  == "#") && (\$72 != "0") && (\$74 != "0") ? "pau" : (\$79 == "1") ? \$84 : phonemap(\$63)
#  p.name
    printf "~%s", \$1  == "0" ? "xx" : (\$1  == "#") && (\$73 != "0") && (\$75 != "0") ? "pau" : (\$80 == "1") ? \$85 : phonemap(\$1)
#  c.name
    printf "-%s", (\$2  == "#") && (\$74 != "0") && (\$76 != "0") ? "pau" : (\$81 == "1") ? \$86 : phonemap(\$2)
#  n.name
    printf "+%s", \$3  == "0" ? "xx" : (\$3  == "#") && (\$75 != "0") && (\$77 != "0") ? "pau" : (\$82 == "1") ? \$87 : phonemap(\$3)
#  nn.name
    printf "=%s", \$64 == "0" ? "xx" : (\$64  == "#") && (\$76 != "0") && (\$78 != "0") ? "pau" : (\$83 == "1") ? \$88 : phonemap(\$64)

#  position in syllable (segment)
    printf ":"
    printf "%s",  \$2 == "#" ? "xx" : \$4 + 1
    printf "_%s", \$2 == "#" ? "xx" : \$12 - \$4

##############################
###  SYLLABLE

## previous syllable

#  p.stress
    printf "/A/%s", \$2 == "#" ? \$49 : \$5
#  p.accent
    printf "_%s", \$2 == "#" ? \$51 : \$8
#  p.length
    printf "_%s", \$2 == "#" ? \$53 : \$11

## current syllable

#  c.stress
    printf "/B/%s", \$2 == "#" ? "xx" : \$6
#  c.accent
    printf "-%s", \$2 == "#" ? "xx" : \$9
#  c.length
    printf "-%s", \$2 == "#" ? "xx" : \$12

#  position in word (syllable)
    printf ":%s", \$2 == "#" ? "xx" : \$14 + 1
    printf "-%s", \$2 == "#" ? "xx" : \$30 - \$14

#  position in phrase (syllable)
    printf "&%s", \$2 == "#" ? "xx" : \$15 + 1
    printf "-%s", \$2 == "#" ? "xx" : \$16 + 1

#  position in phrase (stressed syllable)
    printf "#%s", \$2 == "#" ? "xx" : \$17 + 1
    printf "-%s", \$2 == "#" ? "xx" : \$18 + 1

#  position in phrase (accented syllable)
    printf  "\$"
    printf "%s", \$2 == "#" ? "xx" : \$19 + 1
    printf "-%s", \$2 == "#" ? "xx" : \$20 + 1

#  distance from stressed syllable
    printf ">%s", \$2 == "#" ? "xx" : \$21
    printf "-%s", \$2 == "#" ? "xx" : \$22

#  distance from accented syllable 
    printf "<%s", \$2 == "#" ? "xx" : \$23
    printf "-%s", \$2 == "#" ? "xx" : \$24

#  name of the vowel of current syllable
    printf "|%s", \$2 == "#" ? "xx" : phonemap(\$25)

## next syllable

#  n.stress
    printf "/C/%s", \$2 == "#" ? \$50 : \$7
#  n.accent
    printf "+%s", \$2 == "#" ? \$52 : \$10
#  n.length
    printf "+%s", \$2 == "#" ? \$54 : \$13

##############################
#  WORD

##################
## previous word

#  p.gpos
    printf "/D/%s", \$2 == "#" ? \$55 : \$26
#  p.lenght (syllable)
    printf "_%s", \$2 == "#" ? \$57 : \$29

#################
## current word

#  c.gpos
    printf "/E/%s", \$2 == "#" ? "xx" : \$27
#  c.lenght (syllable)
    printf "+%s", \$2 == "#" ? "xx" : \$30

#  position in phrase (word)
    printf ":%s", \$2 == "#" ? "xx" : \$32 + 1
    printf "+%s", \$2 == "#" ? "xx" : \$33

#  position in phrase (content word)
    printf "&%s", \$2 == "#" ? "xx" : \$34 + 1
    printf "+%s", \$2 == "#" ? "xx" : \$35

#  distance from content word in phrase
    printf "#%s", \$2 == "#" ? "xx" : \$36
    printf "+%s", \$2 == "#" ? "xx" : \$37

##############
## next word

#  n.gpos
    printf "/F/%s", \$2 == "#" ? \$56 : \$28
#  n.lenghte (syllable)
    printf "_%s", \$2 == "#" ? \$58 : \$31

##############################
#  PHRASE

####################
## previous phrase

#  length of previous phrase (syllable)
    printf "/G/%s", \$2 == "#" ? \$59 : \$38

#  length of previous phrase (word)
    printf "_%s"  , \$2 == "#" ? \$61 : \$41

####################
## current phrase

#  length of current phrase (syllable)
    printf "/H/%s", \$2 == "#" ? "xx" : \$39

#  length of current phrase (word)
    printf "=%s",   \$2 == "#" ? "xx" : \$42

#  position in major phrase (phrase)
    printf ":";
    printf "%s", \$44 + 1
    printf "=%s", \$48 - \$44

#  type of tobi endtone of current phrase
    printf "&%s",  \$45

####################
## next phrase

#  length of next phrase (syllable)
    printf "/I/%s", \$2 == "#" ? \$60 : \$40

#  length of next phrase (word)
    printf "_%s",   \$2 == "#" ? \$62 : \$43

##############################
#  UTTERANCE

#  length (syllable)
    printf "/J/%s", \$46

#  length (word)
    printf "+%s", \$47

#  length (phrase)
    printf "-%s", \$48

    printf "\n"
}

function phonemap(string, retstring)
{
 restring = ""
 pattern = "^aa\$|^ou\$|^oo\$|^ei\$|^@@r\$|^eir\$|^ur\$|^aer\$|^owr\$"
 if (match(string,pattern)>0){
   retstring = string "1" 
 }else if (string == "ii"){
   retstring = "iy"
 }else if (string == "uu"){
   retstring = "uw"
 }else if (string == "o"){
   retstring = "aa1"
 }else{
   retstring = string
 }
 return retstring
}
EOF



cat <<EOF >! $tmp/label-mono.awk
{
##############################
###  SEGMENT

#  boundary
    printf "%10.0f %10.0f ", 1e7 * \$65, 1e7 * \$66

#  c.name
    printf "%s", (\$2  == "#") && (\$74 != "0") && (\$76 != "0") ? "pau" : (\$81 == "1") ? \$86 : phonemap(\$2)
    printf "\n"
}

function phonemap(string, retstring)
{
 restring = ""
 pattern = "^aa\$|^ou\$|^oo\$|^ei\$|^@@r\$|^eir\$|^ur\$|^aer\$|^owr\$"
 if (match(string,pattern)>0){
   retstring = string "1" 
 }else if (string == "ii"){
   retstring = "iy"
 }else if (string == "uu"){
   retstring = "uw"
 }else if (string == "o"){
   retstring = "aa1"
 }else{
   retstring = string
 }
 return retstring
}
EOF



# convert utt -> lab
foreach utt ($src/*.utt)
   set base = `basename $utt .utt`
   $dumpfeats -eval     $scpdir/extra_feats_unilex.scm \
              -relation Segment                 \
              -feats    $tmp/label.feats     \
              -output   $tmp/%s.tmp                  \
              $utt
   
   awk -f $tmp/label-full.awk $tmp/$utt:t:r.tmp >! $full/$utt:t:r.lab
   awk -f $tmp/label-mono.awk $tmp/$utt:t:r.tmp >! $mono/$utt:t:r.lab

   rm $tmp/$utt:t:r.tmp
end

# Executed when reached.
int:
    rm -rf "$tmp"

