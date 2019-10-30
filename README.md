# pytorch-database-prep
Scripts to aid in the setup of various databases for pytorch

For speech corpora such as WSJ, these "borrow" from
[Kaldi](http://kaldi-asr.org/). Kaldi's license file has been copied to
``COPYING_kaldi``. Kaldi is Apache 2.0 licensed, as is this repo.

`unlzw.py` is taken, with its original copyright, from the
[unlzw repo](https://github.com/umeat/unlzw).


## Differences from Kaldi

``prune-lm``, provided by [IRSTLM](https://hlt-mt.fbk.eu/technologies/irstlm)
and used in Kaldi, prunes less useful n-grams using a "weighted difference
method." Our own rolled pruning algorithm uses relative entropy threshold,
similar to [SRILM](http://www.speech.sri.com/projects/srilm/). If the threshold
is chosen appropriately, the results
[are](microsoft.com/en-us/research/wp-content/uploads/2000/10/2000-joshuago-icslp.pdf)
[very](https://arxiv.org/pdf/cs/0006025.pdf) similar.