## Setup

```bash
clang -O3 --target=wasm32 --no-standard-libraries -Wl,--export-all -Wl,--no-entry -o mandelbrot.wasm mandelbrot.c
```
