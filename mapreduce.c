#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <math.h>
#include <stddef.h>

#define size 16

struct Pair {
    int key;
    int value;
};

char processor_name[MPI_MAX_PROCESSOR_NAME];
int name_len;

#define ROWS 16
#define COLS 16

// Reads a matrix from a file and stores it in a 1D array in row-major order
void readMatrix(char *filename, int matrix[]) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file %s.\n", filename);
        exit(1);
    }
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            fscanf(file, "%d", &matrix[i*COLS + j]);
        }
    }
    fclose(file);
}

void readMatrixB(char *filename, int matrix[][size]) {

    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file %s.\n", filename);
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fscanf(file, "%d", &matrix[i][j]);
        }
    }
    fclose(file);
}

void standard_multi_matrices(int matrixA[][size], int matrixB[][size], int matrixC[][size]) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrixC[i][j] = 0;
            for (int k = 0; k < size; ++k) {
                matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
}

int main(int argc, char *argv[]) 
{
    int rank, num_processes, num_mappers, num_reducers;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Get_processor_name(processor_name, &name_len);

    srand(time(NULL));

    // Storing Array in Row Major Order
    int matrixA[size * size];

    int matrixB[size][size];
    int matrixC[size][size];
    
    // Reading Matrices
    readMatrix("/mirror/matrix_a.txt", matrixA);
    readMatrixB("/mirror/matrix_b.txt", matrixB);

    // Pair Struct DataType
    int blocklengths[2] = {1, 1};
    MPI_Datatype types[2] = {MPI_INT, MPI_INT};
    MPI_Aint offsets[2];

    offsets[0] = offsetof(struct Pair, key);
    offsets[1] = offsetof(struct Pair, value);

    MPI_Datatype pairType;
    MPI_Type_create_struct(2, blocklengths, offsets, types, &pairType);
    MPI_Type_commit(&pairType);
    
    //Master Process
    if (rank == 0)
    {
        printf("Master with Process Id: %d Running on %s \n",0,processor_name);

        int mappers = num_processes - 3;
        int row_subgrid = size/mappers;

        int start,end,offset = 0;

        int keys[size*size];

        for (int i = 0; i < size*size; i++)
        {
            keys[i] = i+1;
        }

        int store_row[mappers];

        MPI_Request send_reqs[mappers*4];
        int req_index_start = 0;
        int req_index_end = 3;
        int off = 0;

        // Sending RowIndex to Mappers
        for (int i = 1; i <= mappers; i++)
        {   
            printf("Task Map assigned to process %d\n",i);

            start = offset;
            end = start + row_subgrid;


            if (i == mappers)
            {
                end = size;
            }

            store_row[i-1] = end - start;
            int* subgrid = (int*)malloc((end-start)*size*sizeof(int));
            int* subkey = (int*)malloc((end-start)*size*sizeof(int));

            int k = 0;
            for (int x = start; x < end; x++) {
                for (int j = 0; j < 16; j++) {
                    subgrid[k] = matrixA[x * 16 + j];
                    subkey[k] = keys[x * 16 + j];
                    k++;
                }
            }   
            
            int len = (end-start )* size;
            
            MPI_Send(&start, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&end, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(subgrid,len,MPI_INT,i,0,MPI_COMM_WORLD);
            MPI_Send(subkey,len,MPI_INT,i,0,MPI_COMM_WORLD);
            
            offset = end;            
        }

        
        int total_len = size * size * 2;
        struct Pair all_pairs[total_len];
        int index = 0;

        // Receieving Key Value Pairs from mappers
        for (int f = 1; f <= mappers; f++)
        {   
            int len = store_row[f-1] * 2 * size;
            struct Pair received[len];

            MPI_Recv(received, len, pairType, f, f, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
            for (int s = 0; s < len; s++)
            {
                all_pairs[index] = received[s];
                index++;
            }   
        }
        
        //Shuffle the Key-Value Pairs
        int reducers = num_processes-1 - mappers;

        int all_proc[num_processes-1];

        for (int p = 0; p < num_processes-1; p++)
        {
            all_proc[p] = p+1;
        }
        
        int split = total_len/reducers;

        int st = 0;
        int ed = split;
        int pair_index = 0;

        // Sending Pairs to Reducers
        for (int r = num_processes-reducers-1; r < num_processes-1; r++)
        {
            int proc = all_proc[r];
            struct Pair partial_pair[split];
            
            for (int a = 0; a < split; a++)
            {
                partial_pair[a] = all_pairs[pair_index];
                pair_index++;
            }

            MPI_Send(&split,1,MPI_INT,proc,10,MPI_COMM_WORLD);
            MPI_Send(partial_pair,split,pairType,proc,10,MPI_COMM_WORLD);

            printf("Task Reduce assigned to process %d\n",proc);
        }

        // Recieveing Output from Reducers
        int final_output[size*size];

        int temp[512];
        MPI_Recv(temp,split,pairType,6,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

        int partial_group2[split/2];
        MPI_Recv(partial_group2,split/2,pairType,7,12,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

        int final_index = 0;
        for (int q = 0; q < split/2; q++)
        {
            final_output[final_index] = temp[q];
            final_index++;
        }

        for (int q = 0; q < split/2; q++)
        {
            final_output[final_index] = partial_group2[q];
            final_index++;
        }        
        
        int final_2d[size][size]; // assume this is the output 2D array

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) 
            {
                final_2d[i][j] = final_output[i * 16 + j];
            }
        }

        int matA[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) 
            {
                matA[i][j] = matrixA[i * 16 + j];
            }
        }

        standard_multi_matrices(matA,matrixB,matrixC);

        printf("Job Has been completed!\n");

        int flag = 0; 
        // Matching the Matrices
        for (int l = 0; l < size; l++)
        {
            for (int r = 0; r < size; r++)
            {
                if(final_2d[l][r] != matrixC[l][r])
                {
                    flag = 1;
                    break;
                }
            }
        }

        if (flag == 0)
        {
            printf("Matrix Comparison Function Returned: True\n");
        }
        else
        {
            printf("Matrix Comparison Function Returned: False\n");
        }

        //Writing to file
        FILE* fp = fopen("final.txt", "w");
        if (fp == NULL) {
            printf("Error opening file for writing.\n");
        }

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                fprintf(fp, "%d ", final_2d[i][j]);
            }
            fprintf(fp, "\n");
        }

        fclose(fp);
        
    }
    else if ( rank > 0 && rank < num_processes - 2) //Mappers
    {
        MPI_Request req1,req2,req3,req4;
        MPI_Status stat1,stat2,stat3,stat4;   

        int start, end, len;

        MPI_Recv(&start, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&end, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        len = (end - start) * size;
        int rows = (end-start);

        int* grid = (int*)malloc(len * sizeof(int));
        int* key = (int*)malloc(len * sizeof(int));

        MPI_Recv(grid, len, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(key, len, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);

        printf("Process %d received task Map on %s\n",rank,processor_name);

        // Converting back to 2d Array        
        int** grid_2d = (int**)malloc(rows * sizeof(int*));
        for (int i = 0; i < end-start; i++) {
            grid_2d[i] = (int*)malloc(size * sizeof(int));
        }

        int index = 0;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < size; j++)
            {
                grid_2d[i][j] = grid[index];
                index+=1;
            }
        }

        // Converting Key back to 2d Array        
        int** key_2d = (int**)malloc(rows * sizeof(int*));
        for (int i = 0; i < end-start; i++) {
            key_2d[i] = (int*)malloc(size * sizeof(int));
        }

        index = 0;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < size; j++)
            {
                key_2d[i][j] = key[index];
                index+=1;
            }
        }

        int key_len = rows*2*size;
        struct Pair pr[key_len];
        int key_index = 0;

        // Generating Key-Value Pairs

        for (int i = 0; i < rows; i++)
        {
            int index = 0;
            int col_index = 0;

            // Get the current row
            int current_row[size];
            int col = 0;
            for (int x = 0; x < size; x++)
            {
                current_row[x] = grid_2d[i][col];
                col+=1;
            }

            //Get Current Position
            int current_pos[size];  
            col = 0;
            for (int x = 0; x < size; x++)
            {
                current_pos[x] = key_2d[i][col];
                col+=1;
            }


            // Multiply with Matix B's Column
            for (int j = 0; j < size; j++) //Column
            {   
                int partial_sum = 0;
                for (int x = 0; x < size; x++) //Row
                {

                    partial_sum += current_row[col_index] * matrixB[x][j];

                    if (x == (size/2))  //Mid Iteration
                    {
                        int key = current_pos[index];

                        pr[key_index].key = key;
                        pr[key_index].value = partial_sum;
                        key_index+=1;

                        partial_sum = 0;
                    }
                    else if (x == (size-1)) //Last Iteration
                    {
                        int key = current_pos[index];
                        pr[key_index].key = key;
                        pr[key_index].value = partial_sum;
                        key_index+=1;
                        partial_sum = 0;
                    }

                    col_index+=1;
                }

                index += 1;
                col_index = 0;
            }
        }

        MPI_Send(pr,key_len,pairType,0,rank,MPI_COMM_WORLD);

        printf("Process %d has completed task Map\n",rank);
    
    }
    else if (rank >= num_processes-3)   //Reducers
    {
        int len = 0;
        MPI_Recv(&len, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        struct Pair partial_pair[len];
        MPI_Recv(partial_pair, len, pairType, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        printf("Process %d received task Reduce on %s\n",rank,processor_name);

        int group[len/2];   //128

        for (int i = 0; i < len/2; i++)
        {
            group[i] = 0;
        }
        
        for (int i = 0; i < len; i++)
        {
            if (rank == 6)
            {
                group[partial_pair[i].key-1] = group[partial_pair[i].key-1] + partial_pair[i].value;
            }
            else
            {
                group[partial_pair[i].key-129] = group[partial_pair[i].key-129] + partial_pair[i].value;
            }
            
        }
        
        MPI_Send(group,len/2,pairType,0,rank+5,MPI_COMM_WORLD);

        printf("Process %d has completed task Reduce\n",rank);
    }
    

    MPI_Finalize();    
}
