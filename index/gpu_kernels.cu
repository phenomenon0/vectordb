// CUDA kernels for vector distance computation
// Build with: nvcc -c gpu_kernels.cu -o gpu_kernels.o

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

// CUDA kernel for computing vector norms (L2)
__global__ void compute_norms_kernel(
    const float* vectors,
    int num_vectors,
    int dim,
    float* norms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_vectors) {
        float sum = 0.0f;
        const float* vec = vectors + idx * dim;

        for (int i = 0; i < dim; i++) {
            float val = vec[i];
            sum += val * val;
        }

        norms[idx] = sqrtf(sum);
    }
}

// CUDA kernel for computing cosine distances
__global__ void cosine_distance_kernel(
    const float* query_dots,     // [num_queries, num_vectors]
    const float* query_norms,    // [num_queries]
    const float* vector_norms,   // [num_vectors]
    int num_queries,
    int num_vectors,
    float* distances             // [num_queries, num_vectors]
) {
    int q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int v_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (q_idx < num_queries && v_idx < num_vectors) {
        int idx = q_idx * num_vectors + v_idx;

        float dot = query_dots[idx];
        float q_norm = query_norms[q_idx];
        float v_norm = vector_norms[v_idx];

        // Avoid division by zero
        if (q_norm < 1e-9f || v_norm < 1e-9f) {
            distances[idx] = 1.0f;
        } else {
            // Cosine distance = 1 - cosine similarity
            float similarity = dot / (q_norm * v_norm);
            // Clamp to [-1, 1] to handle numerical errors
            similarity = fminf(1.0f, fmaxf(-1.0f, similarity));
            distances[idx] = 1.0f - similarity;
        }
    }
}

// CUDA kernel for computing squared euclidean distances
// Uses the identity: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*dot(a, b)
__global__ void euclidean_distance_kernel(
    const float* query_dots,     // [num_queries, num_vectors] = -2*dot(q, v)
    const float* query_norms_sq, // [num_queries] = ||q||^2
    const float* vector_norms_sq,// [num_vectors] = ||v||^2
    int num_queries,
    int num_vectors,
    float* distances             // [num_queries, num_vectors]
) {
    int q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int v_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (q_idx < num_queries && v_idx < num_vectors) {
        int idx = q_idx * num_vectors + v_idx;

        float neg_2dot = query_dots[idx];
        float q_norm_sq = query_norms_sq[q_idx];
        float v_norm_sq = vector_norms_sq[v_idx];

        // ||a - b||^2 = ||a||^2 + ||b||^2 - 2*dot(a, b)
        float dist_sq = q_norm_sq + v_norm_sq + neg_2dot;

        // Handle numerical errors
        dist_sq = fmaxf(0.0f, dist_sq);

        distances[idx] = sqrtf(dist_sq);
    }
}

// Wrapper functions callable from Go via CGO

extern "C" {

// Initialize CUDA (called once)
void cuda_init(int* status) {
    cudaError_t err = cudaSetDevice(0);
    *status = (err == cudaSuccess) ? 0 : -1;
}

// Compute cosine distances using cuBLAS for dot products
void cuda_cosine_distances(
    const float* queries,
    int num_queries,
    int dim,
    const float* vectors,
    int num_vectors,
    float* distances,
    int* status
) {
    cudaError_t err;
    cublasStatus_t cublas_status;
    cublasHandle_t handle;

    // Initialize cuBLAS
    cublas_status = cublasCreate(&handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        *status = -1;
        return;
    }

    // Allocate device memory
    float *d_queries, *d_vectors, *d_dots;
    float *d_query_norms, *d_vector_norms, *d_distances;

    err = cudaMalloc(&d_queries, num_queries * dim * sizeof(float));
    err = cudaMalloc(&d_vectors, num_vectors * dim * sizeof(float));
    err = cudaMalloc(&d_dots, num_queries * num_vectors * sizeof(float));
    err = cudaMalloc(&d_query_norms, num_queries * sizeof(float));
    err = cudaMalloc(&d_vector_norms, num_vectors * sizeof(float));
    err = cudaMalloc(&d_distances, num_queries * num_vectors * sizeof(float));

    if (err != cudaSuccess) {
        cublasDestroy(handle);
        *status = -2;
        return;
    }

    // Copy data to device
    cudaMemcpy(d_queries, queries, num_queries * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vectors, vectors, num_vectors * dim * sizeof(float), cudaMemcpyHostToDevice);

    // Compute dot products using cuBLAS
    // C = A * B^T where A is queries[num_queries, dim], B is vectors[num_vectors, dim]
    float alpha = 1.0f;
    float beta = 0.0f;

    cublas_status = cublasSgemm(
        handle,
        CUBLAS_OP_T,      // Transpose B (vectors)
        CUBLAS_OP_N,      // Don't transpose A (queries)
        num_vectors,      // rows of B^T
        num_queries,      // rows of A
        dim,              // cols of A = rows of B
        &alpha,
        d_vectors, dim,   // B is [num_vectors, dim] in row-major
        d_queries, dim,   // A is [num_queries, dim] in row-major
        &beta,
        d_dots, num_vectors
    );

    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_queries);
        cudaFree(d_vectors);
        cudaFree(d_dots);
        cudaFree(d_query_norms);
        cudaFree(d_vector_norms);
        cudaFree(d_distances);
        cublasDestroy(handle);
        *status = -3;
        return;
    }

    // Compute norms
    int threads_per_block = 256;
    int blocks_queries = (num_queries + threads_per_block - 1) / threads_per_block;
    int blocks_vectors = (num_vectors + threads_per_block - 1) / threads_per_block;

    compute_norms_kernel<<<blocks_queries, threads_per_block>>>(
        d_queries, num_queries, dim, d_query_norms
    );
    compute_norms_kernel<<<blocks_vectors, threads_per_block>>>(
        d_vectors, num_vectors, dim, d_vector_norms
    );

    // Compute cosine distances
    dim3 block_dim(16, 16);
    dim3 grid_dim(
        (num_vectors + block_dim.x - 1) / block_dim.x,
        (num_queries + block_dim.y - 1) / block_dim.y
    );

    cosine_distance_kernel<<<grid_dim, block_dim>>>(
        d_dots, d_query_norms, d_vector_norms,
        num_queries, num_vectors, d_distances
    );

    // Copy results back
    cudaMemcpy(distances, d_distances, num_queries * num_vectors * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_queries);
    cudaFree(d_vectors);
    cudaFree(d_dots);
    cudaFree(d_query_norms);
    cudaFree(d_vector_norms);
    cudaFree(d_distances);
    cublasDestroy(handle);

    *status = 0;
}

// Compute euclidean distances using cuBLAS for dot products
void cuda_euclidean_distances(
    const float* queries,
    int num_queries,
    int dim,
    const float* vectors,
    int num_vectors,
    float* distances,
    int* status
) {
    cudaError_t err;
    cublasStatus_t cublas_status;
    cublasHandle_t handle;

    cublas_status = cublasCreate(&handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        *status = -1;
        return;
    }

    // Allocate device memory
    float *d_queries, *d_vectors, *d_neg_2dots;
    float *d_query_norms_sq, *d_vector_norms_sq, *d_distances;

    err = cudaMalloc(&d_queries, num_queries * dim * sizeof(float));
    err = cudaMalloc(&d_vectors, num_vectors * dim * sizeof(float));
    err = cudaMalloc(&d_neg_2dots, num_queries * num_vectors * sizeof(float));
    err = cudaMalloc(&d_query_norms_sq, num_queries * sizeof(float));
    err = cudaMalloc(&d_vector_norms_sq, num_vectors * sizeof(float));
    err = cudaMalloc(&d_distances, num_queries * num_vectors * sizeof(float));

    if (err != cudaSuccess) {
        cublasDestroy(handle);
        *status = -2;
        return;
    }

    cudaMemcpy(d_queries, queries, num_queries * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vectors, vectors, num_vectors * dim * sizeof(float), cudaMemcpyHostToDevice);

    // Compute -2 * dot products
    float alpha = -2.0f;
    float beta = 0.0f;

    cublas_status = cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_vectors, num_queries, dim,
        &alpha,
        d_vectors, dim,
        d_queries, dim,
        &beta,
        d_neg_2dots, num_vectors
    );

    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_queries);
        cudaFree(d_vectors);
        cudaFree(d_neg_2dots);
        cudaFree(d_query_norms_sq);
        cudaFree(d_vector_norms_sq);
        cudaFree(d_distances);
        cublasDestroy(handle);
        *status = -3;
        return;
    }

    // Compute squared norms
    int threads_per_block = 256;
    int blocks_queries = (num_queries + threads_per_block - 1) / threads_per_block;
    int blocks_vectors = (num_vectors + threads_per_block - 1) / threads_per_block;

    // Compute norms, then square them
    float *d_query_norms, *d_vector_norms;
    cudaMalloc(&d_query_norms, num_queries * sizeof(float));
    cudaMalloc(&d_vector_norms, num_vectors * sizeof(float));

    compute_norms_kernel<<<blocks_queries, threads_per_block>>>(
        d_queries, num_queries, dim, d_query_norms
    );
    compute_norms_kernel<<<blocks_vectors, threads_per_block>>>(
        d_vectors, num_vectors, dim, d_vector_norms
    );

    // Square the norms (simple element-wise operation on GPU)
    // For simplicity, just copy to host and square (can optimize later)
    float* h_query_norms = (float*)malloc(num_queries * sizeof(float));
    float* h_vector_norms = (float*)malloc(num_vectors * sizeof(float));
    cudaMemcpy(h_query_norms, d_query_norms, num_queries * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vector_norms, d_vector_norms, num_vectors * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_queries; i++) {
        h_query_norms[i] = h_query_norms[i] * h_query_norms[i];
    }
    for (int i = 0; i < num_vectors; i++) {
        h_vector_norms[i] = h_vector_norms[i] * h_vector_norms[i];
    }

    cudaMemcpy(d_query_norms_sq, h_query_norms, num_queries * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_norms_sq, h_vector_norms, num_vectors * sizeof(float), cudaMemcpyHostToDevice);

    free(h_query_norms);
    free(h_vector_norms);
    cudaFree(d_query_norms);
    cudaFree(d_vector_norms);

    // Compute euclidean distances
    dim3 block_dim(16, 16);
    dim3 grid_dim(
        (num_vectors + block_dim.x - 1) / block_dim.x,
        (num_queries + block_dim.y - 1) / block_dim.y
    );

    euclidean_distance_kernel<<<grid_dim, block_dim>>>(
        d_neg_2dots, d_query_norms_sq, d_vector_norms_sq,
        num_queries, num_vectors, d_distances
    );

    // Copy results back
    cudaMemcpy(distances, d_distances, num_queries * num_vectors * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_queries);
    cudaFree(d_vectors);
    cudaFree(d_neg_2dots);
    cudaFree(d_query_norms_sq);
    cudaFree(d_vector_norms_sq);
    cudaFree(d_distances);
    cublasDestroy(handle);

    *status = 0;
}

// Compute dot products (for sparse vectors later)
void cuda_dot_products(
    const float* queries,
    int num_queries,
    int dim,
    const float* vectors,
    int num_vectors,
    float* results,
    int* status
) {
    cublasHandle_t handle;
    cublasStatus_t cublas_status = cublasCreate(&handle);

    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        *status = -1;
        return;
    }

    float *d_queries, *d_vectors, *d_results;

    cudaMalloc(&d_queries, num_queries * dim * sizeof(float));
    cudaMalloc(&d_vectors, num_vectors * dim * sizeof(float));
    cudaMalloc(&d_results, num_queries * num_vectors * sizeof(float));

    cudaMemcpy(d_queries, queries, num_queries * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vectors, vectors, num_vectors * dim * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublas_status = cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_vectors, num_queries, dim,
        &alpha,
        d_vectors, dim,
        d_queries, dim,
        &beta,
        d_results, num_vectors
    );

    cudaMemcpy(results, d_results, num_queries * num_vectors * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_queries);
    cudaFree(d_vectors);
    cudaFree(d_results);
    cublasDestroy(handle);

    *status = (cublas_status == CUBLAS_STATUS_SUCCESS) ? 0 : -3;
}

} // extern "C"
