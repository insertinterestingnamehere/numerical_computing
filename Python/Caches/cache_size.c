#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void l1_l2cache(int *arr, int lenMod, int steps, int cache_line) {
    int i;
    for(i=0; i < steps; ++i) {
        arr[(i*cache_line) & lenMod]++;
    }
}

int main(int argc, char *argv[])
{
    clock_t t1, t2;
    int cache_line = atoi(argv[1]);
    int arr_size = atoi(argv[2]);
    int loops = atoi(argv[3]);

    
    int *arr;
    arr = malloc(arr_size * 1024);
    int lenMod = (arr_size*1024)/(sizeof *arr) - 1;
    
    if (arr != NULL) {
        int steps = 64*1024*1024;
        int i;
        t1 = clock();
        for(i=0; i < loops; ++i) {
            l1_l2cache(arr, lenMod, steps, cache_line);
        }
        t2 = clock();

        free(arr);
        arr = NULL;
        double diff = ((double)t2 - (double)t1)/CLOCKS_PER_SEC/loops;
        printf("%fs per loop\n", diff);
    } else {
        printf("Failed to allocate memory!");
        return 1;
    }
    return 0;
}
