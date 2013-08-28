#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define LEN 128*1024*1024

int main(int argc, char *argv[])
{
    //first argument: step size
    //second argument: number of loops to run
    clock_t t1, t2;
    int loops = atoi(argv[2]);
    int stepsize = atoi(argv[1]);

    int *arr;
    arr = malloc(LEN * sizeof *arr);
    
    if (arr != NULL) {
        int i, j;
        t1 = clock();
        for(i=0; i < loops; ++i) {
            for(j=0; j < LEN; j += stepsize) {
                arr[j] *= 3;
            }
        }
        t2 = clock();

        free(arr);
        arr = NULL;
        double diff = ((double)t2 - (double)t1)/CLOCKS_PER_SEC/loops;
        printf ("%fs per loop\n", diff);
    } else {
        printf("Failed to allocate memory!");
        return 1;
    }
    return 0;
}
