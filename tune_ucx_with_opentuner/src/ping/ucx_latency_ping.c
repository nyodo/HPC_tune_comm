#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

// =================================================================
// MPI Trace Replay & Latency Benchmark (Strict Mode)
// 单位修改为: us (微秒)
// =================================================================

// --- 配置参数 ---
// CSV 文件路径默认值（可通过环境变量 UCX_BENCHMARK_CSV_FILE 覆盖）
#define DEFAULT_CSV_FILENAME "data/processed/misa/16node-64proc-step1-20251203_162824_statistics.csv"
#define MAX_MESSAGE_SIZE (4 * 1024 * 1024) 
#define WARMUP_ITERATIONS 50              
#define BENCHMARK_ITERATIONS 50            
#define CACHE_FLUSH_SIZE (25 * 1024 * 1024)

// *** COMM_TYPE 过滤配置（可通过环境变量 UCX_BENCHMARK_COMM_TYPES 覆盖）***
#define MAX_COMM_TYPES 10
int ALLOWED_COMM_TYPES[MAX_COMM_TYPES];
int NUM_COMM_TYPES = 0;

typedef struct {
    int size;   // 对应 CSV 中的 total_size (Bytes)
    int count;  // 对应 CSV 中的 count
} TraceItem;

// *** 新增: 检查 comm_type 是否在允许列表中 ***
int is_comm_type_allowed(int comm_type) {
    for (int i = 0; i < NUM_COMM_TYPES; i++) {
        if (ALLOWED_COMM_TYPES[i] == comm_type) {
            return 1;
        }
    }
    return 0;
}

// *** 新增: 从环境变量读取配置 ***
void load_config(char** csv_filename) {
    // 1. 读取 CSV 文件路径
    const char* env_csv = getenv("UCX_BENCHMARK_CSV_FILE");
    if (env_csv != NULL) {
        *csv_filename = strdup(env_csv);
    } else {
        *csv_filename = strdup(DEFAULT_CSV_FILENAME);
    }
    
    // 2. 读取允许的 comm_type 列表
    const char* env_comm_types = getenv("UCX_BENCHMARK_COMM_TYPES");
    if (env_comm_types != NULL) {
        // 解析逗号分隔的整数列表，例如 "55,56,57"
        char* types_copy = strdup(env_comm_types);
        char* token = strtok(types_copy, ",");
        NUM_COMM_TYPES = 0;
        
        while (token != NULL && NUM_COMM_TYPES < MAX_COMM_TYPES) {
            ALLOWED_COMM_TYPES[NUM_COMM_TYPES] = atoi(token);
            NUM_COMM_TYPES++;
            token = strtok(NULL, ",");
        }
        free(types_copy);
    } else {
        // 默认值: comm_type = 55
        ALLOWED_COMM_TYPES[0] = 55;
        NUM_COMM_TYPES = 1;
    }
}

// 辅助函数：测量单次操作延迟 (返回单位：秒，保持精度)
double measure_latency(int size, int is_sender, int partner_rank, char* buffer, char* flush_buffer, MPI_Comm comm, int num_pairs) {
    
    MPI_Request request;

    // === 1. Cache 清除 ===
    if (flush_buffer) {
        volatile char sink; 
        for(int i = 0; i < CACHE_FLUSH_SIZE; i++){
            flush_buffer[i] = (char)(i % 128);
        }
        sink = flush_buffer[CACHE_FLUSH_SIZE-1]; 
    }

    MPI_Barrier(comm);

    // === 2. Warmup ===
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        if (is_sender) {
            MPI_Send(buffer, size, MPI_CHAR, partner_rank, 0, comm);
        } else {
            MPI_Irecv(buffer, size, MPI_CHAR, partner_rank, 0, comm, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
    }

    MPI_Barrier(comm);

    double start_time, end_time, local_duration = 0.0;

    // === 3. Timing ===
    if (is_sender) {
        start_time = MPI_Wtime();
        for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
            MPI_Send(buffer, size, MPI_CHAR, partner_rank, 0, comm);
        }
        end_time = MPI_Wtime();
        local_duration = end_time - start_time;
    } else {
        for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
            MPI_Irecv(buffer, size, MPI_CHAR, partner_rank, 0, comm, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
        local_duration = 0.0;
    }

    // === 4. 汇总 ===
    double global_duration_sum = 0.0;
    MPI_Reduce(&local_duration, &global_duration_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    MPI_Barrier(comm);
    usleep(1000 * 10); // 冷却
    MPI_Barrier(comm);

    double avg_latency = 0.0;
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        double avg_total_time = global_duration_sum / num_pairs;
        avg_latency = avg_total_time / BENCHMARK_ITERATIONS;
    }

    MPI_Bcast(&avg_latency, 1, MPI_DOUBLE, 0, comm);
    
    return avg_latency; // 返回秒
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // === 加载配置 (从环境变量) ===
    char* csv_filename = NULL;
    load_config(&csv_filename);

    if (world_size < 2 || world_size % 2 != 0) {
        if (world_rank == 0) fprintf(stderr, "Error: Even number of processes required.\n");
        free(csv_filename);
        MPI_Finalize();
        return 1;
    }

    int half_size = world_size / 2;
    int is_sender = (world_rank < half_size);
    int partner_rank = is_sender ? (world_rank + half_size) : (world_rank - half_size);

    char* buffer = (char*)malloc(MAX_MESSAGE_SIZE);
    char* flush_buffer = (char*)malloc(CACHE_FLUSH_SIZE);

    if (!buffer || !flush_buffer) {
        free(csv_filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    memset(buffer, 0, MAX_MESSAGE_SIZE);

    // --- CSV 读取 ---
    TraceItem* trace_items = NULL;
    int num_items = 0;

    if (world_rank == 0) {
        FILE* fp = fopen(csv_filename, "r");
        if (!fp) {
            fprintf(stderr, "Error: Cannot open file %s\n", csv_filename);
            free(csv_filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        char line[256];
        int capacity = 100;
        trace_items = (TraceItem*)malloc(sizeof(TraceItem) * capacity);

        // Skip header
        fgets(line, sizeof(line), fp);

        while (fgets(line, sizeof(line), fp)) {
            if (num_items >= capacity) {
                capacity *= 2;
                trace_items = (TraceItem*)realloc(trace_items, sizeof(TraceItem) * capacity);
            }
            
            double comm_type_f;
            double size_f;
            int count;
            // 解析: comm_type,index,total_size,count
            // *** 修改: 读取 comm_type 并进行过滤 ***
            if (sscanf(line, "%lf,%*d,%lf,%d", &comm_type_f, &size_f, &count) == 3) {
                int comm_type = (int)comm_type_f;
                
                // *** 过滤: 只处理允许的 comm_type ***
                if (is_comm_type_allowed(comm_type)) {
                    trace_items[num_items].size = (int)size_f;
                    trace_items[num_items].count = count;
                    num_items++;
                }
            }
        }
        fclose(fp);
        
        // *** 修改: 输出过滤信息 ***
        printf(">>> [Rank 0] Loaded %d trace vectors from %s (filtered by COMM_TYPE", num_items, csv_filename);
        for (int i = 0; i < NUM_COMM_TYPES; i++) {
            printf(" %d", ALLOWED_COMM_TYPES[i]);
            if (i < NUM_COMM_TYPES - 1) printf(",");
        }
        printf(").\n");
    }

    // --- 广播 ---
    MPI_Bcast(&num_items, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank != 0) {
        trace_items = (TraceItem*)malloc(sizeof(TraceItem) * num_items);
    }

    int* sizes = (int*)malloc(sizeof(int) * num_items);
    int* counts = (int*)malloc(sizeof(int) * num_items);

    if (world_rank == 0) {
        for(int i=0; i<num_items; i++){
            sizes[i] = trace_items[i].size;
            counts[i] = trace_items[i].count;
        }
    }

    MPI_Bcast(sizes, num_items, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(counts, num_items, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank != 0) {
        for(int i=0; i<num_items; i++){
            trace_items[i].size = sizes[i];
            trace_items[i].count = counts[i];
        }
    }
    
    free(sizes);
    free(counts);

    // --- Replay ---
    double total_simulated_time = 0.0;
    
    for (int i = 0; i < num_items; i++) {
        int msg_size = trace_items[i].size;
        int call_count = trace_items[i].count;

        if (msg_size > MAX_MESSAGE_SIZE) msg_size = MAX_MESSAGE_SIZE;

        // one_op_latency 单位: 秒
        double one_op_latency = measure_latency(msg_size, is_sender, partner_rank, buffer, flush_buffer, MPI_COMM_WORLD, half_size);

        if (world_rank == 0) {
            // 计算逻辑: 单次延迟 * 次数
            double step_time = one_op_latency * call_count;
            total_simulated_time += step_time;
        }
    }

    // --- 输出修改部分 ---
    if (world_rank == 0) {
        printf("=========================================================\n");
        printf(" Trace Replay Summary \n");
        printf(" Vectors Processed: %d\n", num_items);
        printf("---------------------------------------------------------\n");
        
        // 将秒转换为微秒 (us)
        // 1 秒 = 1,000,000 微秒
        double total_time_us = total_simulated_time * 1000000.0;

        printf(" Total Execution Time: %.2f us\n", total_time_us);
        printf("=========================================================\n");
    }

    free(trace_items);
    free(buffer);
    free(flush_buffer);
    free(csv_filename);
    MPI_Finalize();
    return 0;
}
