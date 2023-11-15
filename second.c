// 1000*1000 matrix multiplication
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 1000

int main() {
    int i, j, k;
    int r1 = SIZE, c1 = SIZE, r2 = SIZE, c2 = SIZE;

    // Dynamic memory allocation for matrices
    int **first = (int **)malloc(r1 * sizeof(int *));
    int **second = (int **)malloc(r2 * sizeof(int *));
    int **result = (int **)malloc(r1 * sizeof(int *));
    for (i = 0; i < r1; ++i) {
        first[i] = (int *)malloc(c1 * sizeof(int));
        second[i] = (int *)malloc(c2 * sizeof(int));
        result[i] = (int *)malloc(c2 * sizeof(int));
    }

    // Fill matrices with random integers
    srand(1); // Seed for reproducibility
    for (i = 0; i < r1; ++i) {
        for (j = 0; j < c1; ++j) {
            first[i][j] = rand() % 10;
        }
    }

    for (i = 0; i < r2; ++i) {
        for (j = 0; j < c2; ++j) {
            second[i][j] = rand() % 10;
        }
    }

    // Initialize result matrix for the general case
    for (i = 0; i < r1; ++i) {
        for (j = 0; j < c2; ++j) {
            result[i][j] = 0;
        }
    }

    // Multiplication for general case
    #pragma omp parallel for private(i, j, k) collapse(2)
    for (i = 0; i < r1; ++i) {
        for (j = 0; j < c2; ++j) {
            for (k = 0; k < c1; ++k) {
                result[i][j] += first[i][k] * second[k][j];
            }
        }
    }

    // Print result for the general case
    printf("Result for the general case:\n");
    for (i = 0; i < r1; ++i) {
        for (j = 0; j < c2; ++j) {
            printf("%d ", result[i][j]);
        }
        printf("\n");
    }

    // Free dynamically allocated memory
    for (i = 0; i < r1; ++i) {
        free(first[i]);
        free(second[i]);
        free(result[i]);
    }
    free(first);
    free(second);
    free(result);

    return 0;
}
