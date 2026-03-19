#!/bin/bash
gcc -shared -fPIC -O2 -o integrate_lib.so     integrate_lib.c integrate.c integrate_io.c stats_io.c -lm