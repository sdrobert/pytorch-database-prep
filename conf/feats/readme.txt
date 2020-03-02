Copied in large from my experiments on features
https://github.com/sdrobert/more-or-let, https://github.com/sdrobert/pytorch-kaldi.

These are suggestions, not requirements. The "standard" setup is the
fbank_41.json config.

Kaldi's TIMIT recipe, for some reason, uses the power spectrum instead of the
magnitude spectrum. Kaldi also does dithering and pre-emphasis. We diverge
from the above repositories on these points since we're no longer so focused on
replicating Kaldi.
