Day 27:
This has been one of my toughest kernel implementations so far. I wrote a single fused kernel for scoring all the matched strings and implemented several optimizations, such as using pinned memory and reducing the quadratic memory usage of wavefront parallelism to linear shared memory

https://github.com/AkashKarnatak/fcuk/
