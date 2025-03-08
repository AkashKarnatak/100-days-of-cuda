Day 24
Started working on my fuzzy match library `fcuk` (fuzzy cuda kernel). Implemented a matching and scoring algorithm for strings using dynamic programming similar to wagner fischer's algorithm with affine gap penalty. The scoring system is similar to fzy. Right now I have only implemented the CUDA kernel for the scoring algorithm. Still so many optimizations left.

https://github.com/AkashKarnatak/fcuk/
