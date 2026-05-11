#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int rank, size, N;

    MPI_Init(&argc, &argv);                      
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);        
    MPI_Comm_size(MPI_COMM_WORLD, &size);        

    FILE *fa = NULL, *fb = NULL;
    int *A = NULL, *B = NULL, *C = NULL;

    if (rank == 0) {
        fa = fopen("a.txt", "r");
        fb = fopen("b.txt", "r");

        if (fa == NULL || fb == NULL) {
            printf("Dosya acilamadi!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fscanf(fa, "%d", &N);
        fscanf(fb, "%d", &N);

        A = (int*) malloc(N * N * sizeof(int));
        B = (int*) malloc(N * N * sizeof(int));
        C = (int*) malloc(N * N * sizeof(int));

        for (int i = 0; i < N * N; i++)
            fscanf(fa, "%d", &A[i]);

        for (int i = 0; i < N * N; i++)
            fscanf(fb, "%d", &B[i]);

        fclose(fa);
        fclose(fb);
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (N % size != 0) {
        if (rank == 0)
            printf("Hata: N process sayisina tam bolunmeli!\n");

        MPI_Finalize();
        return 1;
    }

    if (rank != 0) {
        B = (int*) malloc(N * N * sizeof(int));
    }

    MPI_Bcast(B, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    int rows = N / size;

    int *local_A = (int*) malloc(rows * N * sizeof(int));
    int *local_C = (int*) malloc(rows * N * sizeof(int));

    MPI_Scatter(A, rows * N, MPI_INT,
                local_A, rows * N, MPI_INT,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i * N + j] = 0;

            for (int k = 0; k < N; k++) {
                local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    MPI_Gather(local_C, rows * N, MPI_INT,
               C, rows * N, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n=== SONUC MATRISI ===\n");

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                printf("%d ", C[i * N + j]);

            printf("\n");
        }

        printf("\n=== PERFORMANS ===\n");
        printf("Process sayisi: %d\n", size);
        printf("Toplam sure: %f saniye\n", end - start);
    }

    free(A);
    free(B);
    free(C);
    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;
}
