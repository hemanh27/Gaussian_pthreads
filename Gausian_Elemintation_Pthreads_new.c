#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>
#include <limits.h>
#include <unistd.h>
#include <time.h>

#define Maximum_Matrix_Size 2000		// Max value of N
int Matrix_Size;						// Matrix size
int procs;								// Total number of threads
const int MAX_THREADS = 100;			// Maximum number of threads
int NumThreads;							// Number of threads to use

int min(int a, int b)
{ 
	if (a < b)
		 return a;
	else if ( a == b)
		return -1;
	else
		return b;
}
// A * X = B. Solve for X
float A[Maximum_Matrix_Size][Maximum_Matrix_Size];
float B[Maximum_Matrix_Size];
float X[Maximum_Matrix_Size];

pthread_t Threads[_POSIX_THREAD_THREADS_MAX];
pthread_mutex_t Mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t CountLock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t NextIter = PTHREAD_COND_INITIALIZER;

int Norm, CurrentRow, Count;

void create_threads();					// This function spawns off <NumThreads>.
void* gaussPT(void*);					// This function runs concurrently in <NumThreads> threads.
int find_next_row_chunk(int*, int);		// This function determines the chunk size of rows.
void barrier(int*);						// This function implements a barrier synchronisation.
void wait_for_threads();				// This function awaits the termination of all threads.
void initialise_inputs();				// Initialise A, B and X; X is set to 0.0s.
void print_inputs();					// Print input matrices.
void print_X();							// Print solution matrix.

int main()
{
	int row, col;
	Matrix_Size = 144;
	NumThreads = 8;

	printf("Matrix dimension N = %d\n", Matrix_Size);
	printf("Number of threads = %d\n", NumThreads);

	initialise_inputs();
	print_inputs();

	CurrentRow = Norm + 1;
	Count = NumThreads - 1;

//	printf("\nStarting Clock.\n");
	clock_t start = clock(); /*start timer*/

	create_threads();
	wait_for_threads();
	// Diagonal elements are not normalised to 1. This is treated in back substitution.

	/*******************Back substitution.*************************/
	for (row = Matrix_Size - 1; row >= 0; row--)
	{
		X[row] = B[row];
		for (col = Matrix_Size - 1; col > row; col--)
			X[row] -= A[row][col] * X[col];
		X[row] /= A[row][row];
	}

//	printf("Stopped clock.\n");
	clock_t end = clock();
	print_X();

	printf("Elapsed time: %f seconds\n", (double) (end - start) / CLOCKS_PER_SEC);
	return 0;
}

void initialise_inputs()
{
	printf("\nInitialising...\n");
	int col, row;
	for (col = 0; col < Matrix_Size; col++)
	{
		//The addition of 1 is to ensure non-zero entries in the coefficient matrix A.
		for (row = 0; row < Matrix_Size; row++)
			A[row][col] = (float)rand() / RAND_MAX + 1;
		B[col] = (float)rand() / RAND_MAX + 1;
		X[col] = 0.0;
	}
}


void print_inputs()
{
	int row, col;
	if (Matrix_Size < 10)
	{
		printf("\nA = \n\t");
		for (row = 0; row < Matrix_Size; row++)
			for (col = 0; col < Matrix_Size; col++)
				printf("%f6.3 %s", A[row][col], col < (Matrix_Size - 1)?" ":"\n\t");
		printf("\nB = [ ");

		for (col = 0; col < Matrix_Size; col++)
			printf("%f6.3 %s", B[col], col < (Matrix_Size - 1) ? " " : "]\n");
	}
}


void print_X()
{
	int row;
	if (Matrix_Size < 10)
	{
		printf("\nX = [");
		for (row = 0; row < Matrix_Size; row++)
			printf("%f6.3 %s", X[row], row < (Matrix_Size - 1) ? " " : "]\n");
	}
}


void* gaussPT(void* dummy)
{
	int myRow = 0, // <myRow> denotes the first row of the chunk assigned to a thread.
		row, col;
	int myNorm = 0; 	// Normalisation row.

	float multiplier;
	int chunkSize;

	// Gaussian elimination begins here.

	while (myNorm < Matrix_Size - 1)
	{
		// Ascertain the row chunk to be assigned to this thread
		while (chunkSize = find_next_row_chunk(&myRow, myNorm))
		{
			// We perform the eliminations across these rows concurrently.
			for (row = myRow; row < (min(Matrix_Size, myRow + chunkSize)); row++)
			{
				multiplier = A[row][myNorm] / A[myNorm][myNorm];
				for (col = myNorm; col < Matrix_Size; col++)
					A[row][col] -= A[myNorm][col] * multiplier;
				B[row] -= B[myNorm] * multiplier;
			}
		}
		barrier(&myNorm);
	}
	return 0;
}


void barrier(int* myNorm)
{
	pthread_mutex_lock(&CountLock);

	if (Count == 0)
	{
		Norm++;
		Count = NumThreads - 1;
		CurrentRow = Norm + 1;
		pthread_cond_broadcast(&NextIter);
	}
	else
	{
		Count--;
		pthread_cond_wait(&NextIter, &CountLock);
	}
	*myNorm = Norm;
	pthread_mutex_unlock(&CountLock);
}


int find_next_row_chunk(int* myRow, int myNorm)
{
	int chunkSize;
	pthread_mutex_lock(&Mutex);
	*myRow = CurrentRow;
	chunkSize = (*myRow < Matrix_Size) ? (Matrix_Size - myNorm - 1) / (2 * NumThreads) + 1 : 0;
	CurrentRow += chunkSize;
	pthread_mutex_unlock(&Mutex);
	return chunkSize;
}


void create_threads()
{
	int i;
	for (i = 0; i < NumThreads; i++)
		pthread_create(&Threads[i], NULL, gaussPT, NULL);
}


void wait_for_threads()
{
	int i;
	for (i = 0; i < NumThreads; i++)
		pthread_join(Threads[i], NULL);
}
