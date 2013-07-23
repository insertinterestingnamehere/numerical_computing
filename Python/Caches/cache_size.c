#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void l1_l2cache(int *arr, int arrsize, int steps, int cache_line) {
    int lenMod = arrsize - 1;

    for(int i=0; i < steps; ++i) {
        arr[(i*cache_line) & lenMod]++;
    }
}

int kb_arrsize(int kb, int bytelen) {
    return (kb*1024)/bytelen;
}

int main(int argc, char *argv[])
{
    clock_t t1, t2;
    int loops = atoi(argv[3]);
    int arr_size = kb_arrsize(atoi(argv[2]), 8);
    int cache_line = atoi(argv[1]);
    
    int *arr;
    arr = (int *) malloc(arr_size*sizeof(int));

    int steps = 64*1024*1024;
    t1 = clock();
    for(int i=0; i < loops; ++i) {
        l1_l2cache(arr, arr_size, steps, cache_line);
    }
    t2 = clock();

    free(arr);
    double diff = ((double)t2 - (double)t1)/CLOCKS_PER_SEC/loops;
    printf("%fs per loop", diff)
    return 0;
}
