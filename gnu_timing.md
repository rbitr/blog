# Comparing python with GNU/Linux tools 

I always took it as common knowledge that core utilities like `awk` and `sed` are written in optimized c-code with no overhead and run as fast as anything possibly could. I haven't quite let this idea go, but I found an interesting counterexample that makes me feel better about writing python code for basic data manipulations.

## Situation

I have a text file `text_tokens_sci_spacy` where each line is an array of tokens from articles that have been parsed by sci-spacy. (The text is from the CORD-19 collection of coronavirus literature). The file has 66,284 lines and is 53 MB. I want to write a function that gives me the most frequent words after removing stopwords. Each line of the file looks like:

```
["for", "replication", "they", "provide", "genomes", "with"... ]
```

## Analysis

I wrote the following python, without any attempt at optimization:

```python
from collections import Counter
import json
import random
from stopwords import stopwords

def get_sample_tokens(lines,n=10):
    token_dict = Counter([t for line in lines for t in line])
    token_dict = [t for t in token_dict.most_common()]
    token_dict = list(filter(lambda x: x[0] not in stopwords, token_dict))
    return token_dict[0:n]

text_lines_tokenized = []
with open ('scratch/text_tokens_sci_spacy') as f:
    for line in f:
        text_lines_tokenized.append(json.loads(line))

t = get_sample_tokens(text_lines_tokenized,method)

for u in t:
    print(u)

```

Note that the `stopwords` import just declares an array of 127 stopwords (called `stopwords`). An example run is below - the time varies a bit, but it's usually just over two seconds.

```
$ time python topwords.py 
('preprint', 32746)
('cases', 16036)
('data', 15270)
('cells', 15255)
('patients', 14849)
('copyright', 14288)
('holder', 14162)
('license', 14128)
('author/funder', 13945)
('covid-19', 13507)

real	0m2.333s
user	0m2.149s
sys	0m0.184s
```

So I wanted to see how much speed I was leaving on the table by not writing a script to do this. I tried to translate it into a few commands piped together, not optimized, but assuming it would naturally be much faster:

```
$ time cat scratch/text_tokens_sci_spacy | jq '.[]' | tr -d '"' | sort | \
uniq -c | LC_ALL=C sort -nr -k1  | grep -E -v -f scratch/stopwordsrx | head
  32746 preprint
  16036 cases
  15270 data
  15255 cells
  14849 patients
  14288 copyright
  14162 holder
  14128 license
  13945 author/funder
  13507 covid-19

real	0m8.963s
user	0m10.121s
sys	0m0.605s
```

Here, stopwordsrx is the list of stopwords with word boundaries `\b` added to let me use `grep` to filter them out. The good news is that the script is very short and I got to use `jq`. The bad news is that it takes 4 times as long as the python code, and makes me question everything I believe in. 

An obvious thing to try is to count word frequencies directly instead of lazy reliance on the `sort | uniq -c | sort` pattern to get an ordered list of counts:

```
$ time cat scratch/text_tokens_sci_spacy | jq '.[]' | tr -d '"' | \
awk -v "OFS"="," '{word[$0]++} END {for (w in word) print word[w], w}' \
| LC_ALL=C sort -nr -t, -k1  | grep -E -v -f scratch/stopwordsrx | head
32746,preprint
16036,cases
15270,data
15255,cells
14849,patients
14288,copyright
14162,holder
14128,license
13945,author/funder
13507,covid-19

real	0m4.172s
user	0m5.204s
sys	0m0.460s
```

There are 5,729,855 tokens, so unsurprisingly it is much faster when we dispense with the initial sort. But we're still twice as slow as python.

The next obvious target is `jq` which could be considered overkill to remove the commas from between the tokens:

```
$ time cat scratch/text_tokens_sci_spacy | tr -d '[]"' | \
sed -e 's/,[[:space:]]*/\n/g' | awk -v "OFS"="," '{word[$0]++} \
END {for (w in word) print word[w], w}' | LC_ALL=C sort -nr -t, -k1 \
 | grep -E -v -f scratch/stopwordsrx | head
32746,preprint
16036,cases
15270,data
15259,cells
14849,patients
14288,copyright
14162,holder
14128,license
13945,author/funder
13512,covid-19

real	0m2.186s
user	0m2.998s
sys	0m0.292s
```

The code is less elegant (and if you compare the token counts they differ very slightly, presumably because of some errant quotes). But it finally achieves native python performance. 

I played with a few other changes to both the python code and the script (including doing away with the first `cat` that I know is disliked by a subculture within CS) but was unable to reduce the execution time any more.

## Conclusion

One common sense improvement was to avoid sorting a really long array. Writing python, I naturally avoided it because of the options available (a Counter) but with the script I walked right into it. The most disappointing thing was that `jq` added so much overhead. That program was being used for something very simple, so it is not a fair characterization of what it can do, but on the other hand `json.loads` manages to efficiently pull each line into an array. And finally my speedup with `sed` and `tr` was evidently a bit hacky, since it changed the total character counts. However, the dataset is not clean so we expect a bit of 'randomness' with different parsing methods. 

I prefer to use a shell script whenever I can, both because it feels like less work to write one, and I find it appealing to think in terms of pipes. But python is the clear winner here, because it is just as fast, was naturally optimized, and for this case, where my eventual plan is to write a much more complex program, fits with my development goals.
