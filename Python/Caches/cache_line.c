#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define LEN 64*1024*1024*2 //8MB

void cache_line(int *arr, int k) {
    int i;
    for(i=0; i < LEN; i += k) {
        arr[i] *= 3;
    }
}

int main(int argc, char *argv[])
{
    //first argument: step size
    //second argument: number of loops to run
    clock_t t1, t2;
    int loops = atoi(argv[2]);
    int stepsize = atoi(argv[1]);

    int *arr;
    int i;
    arr = (int *) malloc(LEN*sizeof(int));
    t1 = clock();
    for(i=0; i < loops; ++i) {
        cache_line(arr, stepsize);
    }
    t2 = clock();

    free(arr);
    double diff = ((double)t2 - (double)t1)/CLOCKS_PER_SEC/loops;
    printf ("%fs per loop\n", diff);
    return 0;
}
