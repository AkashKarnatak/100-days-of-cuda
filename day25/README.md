Day 25:
Fused compute bonus and scoring kernel and fixed previous logical bugs. Also implemented a matching kernel using parallel reduction with time complexity O(n log m). (where n is pattern len and m is string len) Sequential CPU algorithm is O(n + m) but pattern length is generally small so I am assuming the algo should run practically faster on GPU. Need to benchmark.

https://github.com/AkashKarnatak/fcuk/
