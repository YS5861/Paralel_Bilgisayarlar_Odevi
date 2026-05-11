from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    with open("a.txt") as f:
        N = int(f.readline())
        A = np.loadtxt(f, dtype=int)

    with open("b.txt") as f:
        _ = int(f.readline())
        B = np.loadtxt(f, dtype=int)
else:
    N = None
    A = None
    B = None

# N broadcast
N = comm.bcast(N, root=0)

if rank != 0:
    B = np.zeros((N, N), dtype=int)

comm.Bcast(B, root=0)

rows = N // size
local_A = np.zeros((rows, N), dtype=int)

comm.Scatter(A, local_A, root=0)

comm.Barrier()
start = MPI.Wtime()

# hesaplama
local_C = np.dot(local_A, B)

comm.Barrier()
end = MPI.Wtime()

if rank == 0:
    C = np.zeros((N, N), dtype=int)
else:
    C = None

comm.Gather(local_C, C, root=0)

if rank == 0:
    print("\n=== SONUC MATRISI ===")
    print(C)

    print("\n=== PERFORMANS ===")
    print(f"Process sayisi: {size}")
    print(f"Toplam sure: {end - start:.6f} saniye\n")
