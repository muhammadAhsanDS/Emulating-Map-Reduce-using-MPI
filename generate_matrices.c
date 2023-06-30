#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 16

int main() {
    srand(time(NULL)); // seed random number generator
    
    // create random matrices
    int matrix_a[SIZE][SIZE];
    int matrix_b[SIZE][SIZE];

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix_a[i][j] = (rand() % 100) + 1; // values between 1 and 100
            matrix_b[i][j] = (rand() % 100) + 1;
        }
    }
    
    // write matrices to files
    FILE *file_a = fopen("matrix_a.txt", "w");
    FILE *file_b = fopen("matrix_b.txt", "w");
    
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            fprintf(file_a, "%d ", matrix_a[i][j]);
            fprintf(file_b, "%d ", matrix_b[i][j]);
        }
        fprintf(file_a, "\n");
        fprintf(file_b, "\n");
    }
    fclose(file_a);
    fclose(file_b);
    
    return 0;
}
