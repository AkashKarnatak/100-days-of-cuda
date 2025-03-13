Day 28:

The day of utter disappointment. Today I implemented fused matching kernel and finally decided to benchmark my code. Turns out that the main bottleneck in my cuda implementation is copying data back and forth for processing (which is inevitable) making it look laughable when I compare with CPU implementation for small inputs (2000 to 50k input sources). The speedup is only noticeable when I run the on large number of sources (355k).

Okay let's be a little bit unfair and just compare the kernel execution time with sequential CPU implementation and we get a 5x speedup. Yay!! But when I compare kernel execution time with parallelized CPU implementation (with includes sorting the results as well btw) the times are almost the same (3ms).

Bruh!!

This is so disappointing. I kinda knew this wasn't a problem worth optimizing on CUDA, which I assume is primarily because the subproblem of matching and scoring happens on relatively short strings (length <1024). There just isn’t much computation involved.

It all began when I was going through blink.cmp's docs and I noticed that it used a fuzzy matcher library called frisbee that makes blink.cmp extremely fast. When I checked out Frisbee, I saw that it uses parallelization and SIMD to achieve its speedup, so it made me genuinely curious if I could beat this library with a CUDA implementation.

After all this headache and these disappointing results, I’ll take the L and move on. After all I cannot ignore the CUDA copy time, which makes this library practically unusable. It was a good learning experience and I got to implement whatever I had learned so far plus learn new techniques. Onto something different from tomorrow.

https://github.com/AkashKarnatak/fcuk/
