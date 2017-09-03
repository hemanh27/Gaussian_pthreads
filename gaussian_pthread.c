/*****************************************************
 *
 * Gaussian elimination
 *
 * pthread parallel version
 *
 *****************************************************/
#include <pthread.h>
#include <stdio.h>

#define MAX_SIZE 4096
#define MAX_CPUS 8

typedef double matrix[MAX_SIZE][MAX_SIZE];

int	N;		/* matrix size		*/
int	maxnum;		/* max number of element*/
char	*Init;		/* matrix init type	*/
int	PRINT;		/* print switch		*/
matrix	A;		/* matrix A		*/
double	b[MAX_SIZE];	/* vector b             */
double	y[MAX_SIZE];	/* vector y             */
int nthreads, tid;

pthread_barrier_t thread_barrier;
pthread_mutex_t mutex_division_step = PTHREAD_MUTEX_INITIALIZER;

/* forward declarations */
void work(void *);
void Init_Matrix(void);
void Print_Matrix(void);
void Init_Default(void);
int Read_Options(int, char **);

int
main(int argc, char **argv)
{
    int i, timestart, timeend, iter, thread_num=0;

    pthread_t worker_threads[MAX_CPUS];
    pthread_barrier_init (&thread_barrier, NULL, MAX_CPUS);

    Init_Default();		/* Init default values	*/
    Read_Options(argc,argv);	/* Read arguments	*/
    Init_Matrix();		/* Init the matrix	*/

    for(thread_num = 0; thread_num < MAX_CPUS; thread_num++)
    {
        pthread_create (&worker_threads[thread_num], NULL, (void *) &work, (void *) thread_num);
    }

    for(thread_num = 0; thread_num < MAX_CPUS; thread_num++)
    {
        pthread_join(worker_threads[thread_num],NULL);
    }

    pthread_barrier_destroy(&thread_barrier);

    if (PRINT == 1)
        Print_Matrix();
}

void work(void *thread_num)
{
    int i, j, k;
    int thread_id = (int)thread_num;

	pthread_mutex_init(&mutex_division_step,NULL);

    /* Gaussian elimination algorithm, Algo 8.4 from Grama */
    for (k = 0; k < N; k++)
    {

       if(thread_id == (k % MAX_CPUS)) /* Parallelization at row level */
       {
        /* Outer loop */
        for (j = k+1; j < N; j++)
        {
            A[k][j] = A[k][j] / A[k][k]; /* Division step */
        }

        y[k] = b[k] / A[k][k];
        A[k][k] = 1.0;
        }
       pthread_barrier_wait(&thread_barrier); /* All threads have to wait for before starting of elimination. */

        for (i = k+1; i < N; i++)
        {
            for (j = k+1; j < N; j++)
                A[i][j] = A[i][j] - A[i][k]*A[k][j]; /* Elimination step */

            b[i] = b[i] - A[i][k]*y[k];
            A[i][k] = 0.0;
        }
    }
}

void
Init_Matrix()
{
    int i, j;

    printf("\nsize      = %dx%d ", N, N);
    printf("\nmaxnum    = %d \n", maxnum);
    printf("Init	  = %s \n", Init);
    printf("Initializing matrix...");

    if (strcmp(Init,"rand") == 0)
    {
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                if (i == j) /* diagonal dominance */
                    A[i][j] = (double)(rand() % maxnum) + 5.0;
                else
                    A[i][j] = (double)(rand() % maxnum) + 1.0;
            }
        }
    }
    if (strcmp(Init,"fast") == 0)
    {
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                if (i == j) /* diagonal dominance */
                    A[i][j] = 5.0;
                else
                    A[i][j] = 2.0;
            }
        }
    }

    /* Initialize vectors b and y */
    for (i = 0; i < N; i++)
    {
        b[i] = 2.0;
        y[i] = 1.0;
    }

    printf("done \n\n");
    if (PRINT == 1)
        Print_Matrix();
}

void
Print_Matrix()
{
    int i, j;

    printf("Matrix A:\n");
    for (i = 0; i < N; i++)
    {
        printf("[");
        for (j = 0; j < N; j++)
            printf(" %5.2f,", A[i][j]);
        printf("]\n");
    }
    printf("Vector b:\n[");
    for (j = 0; j < N; j++)
        printf(" %5.2f,", b[j]);
    printf("]\n");
    printf("Vector y:\n[");
    for (j = 0; j < N; j++)
        printf(" %5.2f,", y[j]);
    printf("]\n");
    printf("\n\n");
}

void
Init_Default()
{
    N = 2048;
    Init = "rand";
    maxnum = 15.0;
    PRINT = 0;
}

int
Read_Options(int argc, char **argv)
{
    char    *prog;

    prog = *argv;
    while (++argv, --argc > 0)
        if (**argv == '-')
            switch ( *++*argv )
            {
            case 'n':
                --argc;
                N = atoi(*++argv);
                break;
            case 'h':
                printf("\nHELP: try sor -u \n\n");
                exit(0);
                break;
            case 'u':
                printf("\nUsage: sor [-n problemsize]\n");
                printf("           [-D] show default values \n");
                printf("           [-h] help \n");
                printf("           [-I init_type] fast/rand \n");
                printf("           [-m maxnum] max random no \n");
                printf("           [-P print_switch] 0/1 \n");
                exit(0);
                break;
            case 'D':
                printf("\nDefault:  n         = %d ", N);
                printf("\n          Init      = rand" );
                printf("\n          maxnum    = 5 ");
                printf("\n          P         = 0 \n\n");
                exit(0);
                break;
            case 'I':
                --argc;
                Init = *++argv;
                break;
            case 'm':
                --argc;
                maxnum = atoi(*++argv);
                break;
            case 'P':
                --argc;
                PRINT = atoi(*++argv);
                break;
            default:
                printf("%s: ignored option: -%s\n", prog, *argv);
                printf("HELP: try %s -u \n\n", prog);
                break;
            }
}

