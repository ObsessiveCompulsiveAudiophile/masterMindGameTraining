// Mastermind Deep Q-Network (DQN) Agent - V4 (GPU Accelerated with CUDA) - Patched
// by Gemini
//
// This version ports the entire DQN training and inference logic to the GPU
// using custom CUDA kernels for maximum performance.
// Fixes compilation errors related to CUDART_INF_F and vector<bool>.
//
// How to Compile (using NVIDIA's nvcc):
// nvcc -o Mastermind_DQN_GPU.exe kernel.cu -O3 -arch=sm_75 --std=c++17 -Xcompiler "/O2"
// (Adjust -arch=sm_XX to match your GPU's compute capability, e.g., sm_86 for RTX 30xx)

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath> // For HUGE_VALF
#include <map>
#include <random>
#include <iomanip>
#include <deque>
#include <limits>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA Error Checking Macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// ===================================================================================
// Game & Network Configuration
// ===================================================================================
const int N_POSITIONS = 4;
const int N_COLORS = 6;
const int S_CODES = 1296; // This is our Action Space & part of State Space
const int IMARK0_WIN = 15;

const int NUM_EPISODES = 20000;
const float LEARNING_RATE = 1e-4f;
const float GAMMA = 0.99f;

const int REPLAY_BUFFER_SIZE = 10000;
const int BATCH_SIZE = 64;

const float EPSILON_START = 1.0f;
const float EPSILON_END = 0.01f;
const float EPSILON_DECAY = 0.995f;

const int HIDDEN_SIZE = 256;
const int TARGET_UPDATE_FREQ = 10;

// ===================================================================================
// Game Logic (CPU side)
// ===================================================================================
std::vector<int> h_Valids(S_CODES + 1);
std::vector<int> h_Mark_flat((S_CODES + 1)* (S_CODES + 1));

void FillSetIterative() {
    int idx = 1;
    for (int c1 = 1; c1 <= N_COLORS; ++c1) for (int c2 = 1; c2 <= N_COLORS; ++c2) for (int c3 = 1; c3 <= N_COLORS; ++c3) for (int c4 = 1; c4 <= N_COLORS; ++c4)
        h_Valids[idx++] = c1 * 1000 + c2 * 100 + c3 * 10 + c4;
}

void GenerateMarkTable() {
    std::map<int, int> sign_to_idx;
    int k = 1;
    for (int i = 0; i <= N_POSITIONS; ++i) for (int j = 0; j <= N_POSITIONS; ++j)
        if (i + j <= N_POSITIONS && !(i == N_POSITIONS - 1 && j == 1)) sign_to_idx[i * 10 + j] = k++;
    sign_to_idx[N_POSITIONS * 10] = IMARK0_WIN;

    for (int i = 1; i <= S_CODES; i++) for (int j = i; j <= S_CODES; j++) {
        if (i == j) { h_Mark_flat[i * (S_CODES + 1) + j] = IMARK0_WIN; continue; }
        std::vector<int> a(N_POSITIONS), b(N_POSITIONS);
        int tempA = h_Valids[i], tempB = h_Valids[j];
        for (int k_idx = N_POSITIONS - 1; k_idx >= 0; k_idx--) { a[k_idx] = tempA % 10; b[k_idx] = tempB % 10; tempA /= 10; tempB /= 10; }
        int plus = 0, minus = 0;
        for (int k_idx = 0; k_idx < N_POSITIONS; k_idx++) if (a[k_idx] == b[k_idx]) { plus++; a[k_idx] = 0; b[k_idx] = 99; }
        for (int k_idx = 0; k_idx < N_POSITIONS; k_idx++) for (int l = 0; l < N_POSITIONS; l++) if (a[k_idx] != 0 && a[k_idx] == b[l]) { minus++; b[l] = 99; break; }
        h_Mark_flat[i * (S_CODES + 1) + j] = sign_to_idx.at(plus * 10 + minus);
        h_Mark_flat[j * (S_CODES + 1) + i] = h_Mark_flat[i * (S_CODES + 1) + j];
    }
}


// ===================================================================================
// CUDA Kernels
// ===================================================================================

__global__ void forward_kernel(float* out, float* pre_activation, const float* in, const float* W, const float* b, int batch_size, int input_size, int output_size, bool apply_relu) {
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || out_idx >= output_size) return;

    float sum = 0.0f;
    for (int i = 0; i < input_size; ++i) {
        sum += in[batch_idx * input_size + i] * W[i * output_size + out_idx];
    }
    sum += b[out_idx];

    if (pre_activation != nullptr) {
        pre_activation[batch_idx * output_size + out_idx] = sum;
    }

    if (apply_relu) {
        out[batch_idx * output_size + out_idx] = fmaxf(0.0f, sum);
    }
    else {
        out[batch_idx * output_size + out_idx] = sum;
    }
}


__global__ void calculate_targets_and_loss_delta_kernel(float* delta_out, const float* q_values, const float* next_q_values, const int* actions, const float* rewards, const bool* dones, float gamma, int batch_size, int action_space_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    // Find max Q-value for the next state from the target network
    // FIX #1: Use -HUGE_VALF instead of the non-standard -CUDART_INF_F
    float max_next_q = -HUGE_VALF;
    for (int j = 0; j < action_space_size; ++j) {
        max_next_q = fmaxf(max_next_q, next_q_values[i * action_space_size + j]);
    }

    float target_q = rewards[i] + (dones[i] ? 0.0f : gamma * max_next_q);

    int action_taken = actions[i];
    int q_value_index = i * action_space_size + action_taken;
    float predicted_q = q_values[q_value_index];

    // Initialize all deltas for this sample to 0
    for (int j = 0; j < action_space_size; ++j) {
        delta_out[i * action_space_size + j] = 0.0f;
    }

    // Set the delta only for the action taken. This is the gradient of MSE loss.
    delta_out[q_value_index] = predicted_q - target_q;
}

__global__ void backward_output_layer_kernel(float* dW2, float* db2, float* delta_hidden, const float* delta_out, const float* W2, const float* hidden_activations, const float* pre_activation_hidden, int batch_size, int hidden_size, int output_size) {
    // This kernel calculates gradients for W2, b2 and the error signal for the hidden layer (delta_hidden)

    // Part 1: Calculate delta_hidden (error for the hidden layer)
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int hidden_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && hidden_idx < hidden_size) {
        float error_sum = 0.0f;
        for (int i = 0; i < output_size; ++i) {
            error_sum += delta_out[batch_idx * output_size + i] * W2[hidden_idx * output_size + i];
        }

        // Apply ReLU derivative: if input to ReLU was <= 0, gradient is 0.
        float relu_derivative = (pre_activation_hidden[batch_idx * hidden_size + hidden_idx] > 0.0f) ? 1.0f : 0.0f;
        delta_hidden[batch_idx * hidden_size + hidden_idx] = error_sum * relu_derivative;
    }

    // Use a separate grid-stride loop for updating gradients to avoid launching another kernel
    // Part 2: Calculate gradients dW2 and db2
    int grad_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int items_per_thread = (hidden_size * output_size + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);

    for (int k = 0; k < items_per_thread; ++k) {
        int index = grad_idx * items_per_thread + k;
        if (index >= hidden_size * output_size) continue;

        int h = index / output_size;
        int o = index % output_size;

        float grad_w2 = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            grad_w2 += hidden_activations[b * hidden_size + h] * delta_out[b * output_size + o];
        }
        atomicAdd(&dW2[index], grad_w2);
    }

    if (threadIdx.x < output_size) {
        float grad_b2 = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            grad_b2 += delta_out[b * output_size + threadIdx.x];
        }
        atomicAdd(&db2[threadIdx.x], grad_b2);
    }
}

__global__ void backward_hidden_layer_kernel(float* dW1, float* db1, const float* delta_hidden, const float* states, int batch_size, int state_size, int hidden_size) {
    // This kernel calculates gradients for W1 and b1

    // Part 1: Calculate gradient dW1
    int state_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int hidden_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (state_idx < state_size && hidden_idx < hidden_size) {
        float grad_w1 = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            grad_w1 += states[b * state_size + state_idx] * delta_hidden[b * hidden_size + hidden_idx];
        }
        atomicAdd(&dW1[state_idx * hidden_size + hidden_idx], grad_w1);
    }

    // Part 2: Calculate gradient db1
    if (state_idx == 0 && hidden_idx < hidden_size) { // Only one thread "row" needs to do this
        float grad_b1 = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            grad_b1 += delta_hidden[b * hidden_size + hidden_idx];
        }
        atomicAdd(&db1[hidden_idx], grad_b1);
    }
}


__global__ void sgd_update_kernel(float* weights, float* gradients, float learning_rate, int num_elements, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    // Apply gradient (average over batch) and then reset gradient to 0 for the next iteration
    if (gradients[i] != 0.0f) {
        weights[i] -= learning_rate * (gradients[i] / batch_size);
        gradients[i] = 0.0f;
    }
}

// ===================================================================================
// GPU Network & Agent
// ===================================================================================

struct DeviceQNetwork {
    // Network parameters
    float* W1, * b1, * W2, * b2;
    // Gradients
    float* dW1, * db1, * dW2, * db2;
    // Batch processing buffers
    float* d_states, * d_next_states;
    int* d_actions;
    float* d_rewards;
    bool* d_dones;

    // Intermediate activation/value buffers for training
    float* d_pre_activation_hidden, * d_hidden_activations, * d_q_values;
    float* d_next_q_values;
    float* d_delta_out, * d_delta_hidden;
};


struct Experience {
    std::vector<float> state;
    int action;
    float reward;
    std::vector<float> next_state;
    bool done;
};

class ReplayBuffer {
    std::deque<Experience> memory;
    std::mt19937 rand_gen;
public:
    ReplayBuffer(unsigned int seed) : rand_gen(seed) {}

    void push(const Experience& exp) {
        if (memory.size() >= REPLAY_BUFFER_SIZE) {
            memory.pop_front();
        }
        memory.push_back(exp);
    }

    std::vector<Experience> sample(int batch_size) {
        std::vector<Experience> batch;
        std::vector<int> indices(memory.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rand_gen);
        for (int i = 0; i < batch_size && i < memory.size(); ++i) {
            batch.push_back(memory[indices[i]]);
        }
        return batch;
    }

    int size() const { return memory.size(); }
};

class MastermindEnv {
    int secret_code_idx;
    std::mt19937 rand_gen;
public:
    std::vector<float> state;

    MastermindEnv(unsigned int seed) : rand_gen(seed) { reset(); }

    void reset() {
        std::uniform_int_distribution<int> dist(1, S_CODES);
        secret_code_idx = dist(rand_gen);
        state.assign(S_CODES + 1, 1.0f);
        state[0] = 0.0f; // Index 0 is not used for codes
    }

    bool step(int guess_idx, float& reward) {
        int sign = h_Mark_flat[guess_idx * (S_CODES + 1) + secret_code_idx];
        if (sign == IMARK0_WIN) {
            reward = 20.0f;
            return true;
        }

        int possible_before = 0; for (int i = 1; i <= S_CODES; ++i) if (state[i] > 0.5f) possible_before++;
        for (int i = 1; i <= S_CODES; ++i) {
            if (state[i] > 0.5f && h_Mark_flat[i * (S_CODES + 1) + guess_idx] != sign) {
                state[i] = 0.0f;
            }
        }
        int possible_after = 0; for (int i = 1; i <= S_CODES; ++i) if (state[i] > 0.5f) possible_after++;

        if (possible_after == 0) { reward = -20.0f; return true; }
        reward = (possible_before > possible_after) ? 1.0f : -1.0f;
        return false;
    }
};

class DQNAgent {
public:
    DeviceQNetwork policy_net_gpu, target_net_gpu;
    ReplayBuffer memory;
    float epsilon = EPSILON_START;
    std::mt19937 rand_gen;

    DQNAgent(unsigned int seed) : memory(seed), rand_gen(seed) {}

    void init_gpu_memory() {
        // Initialize weights on CPU first
        std::mt19937 init_rand_gen(rand_gen());
        std::normal_distribution<float> dist1(0.0f, sqrt(2.0f / (S_CODES + HIDDEN_SIZE)));
        std::normal_distribution<float> dist2(0.0f, sqrt(2.0f / (HIDDEN_SIZE + S_CODES)));

        std::vector<float> h_W1(S_CODES * HIDDEN_SIZE), h_b1(HIDDEN_SIZE, 0.0f);
        std::vector<float> h_W2(HIDDEN_SIZE * S_CODES), h_b2(S_CODES, 0.0f);
        for (auto& w : h_W1) w = dist1(init_rand_gen);
        for (auto& w : h_W2) w = dist2(init_rand_gen);

        // Allocate memory for Policy Network
        CHECK_CUDA(cudaMalloc(&policy_net_gpu.W1, S_CODES * HIDDEN_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&policy_net_gpu.b1, HIDDEN_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&policy_net_gpu.W2, HIDDEN_SIZE * S_CODES * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&policy_net_gpu.b2, S_CODES * sizeof(float)));

        // Copy initial weights to GPU
        CHECK_CUDA(cudaMemcpy(policy_net_gpu.W1, h_W1.data(), S_CODES * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(policy_net_gpu.b1, h_b1.data(), HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(policy_net_gpu.W2, h_W2.data(), HIDDEN_SIZE * S_CODES * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(policy_net_gpu.b2, h_b2.data(), S_CODES * sizeof(float), cudaMemcpyHostToDevice));

        // Allocate gradients and zero them out
        CHECK_CUDA(cudaMalloc(&policy_net_gpu.dW1, S_CODES * HIDDEN_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&policy_net_gpu.db1, HIDDEN_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&policy_net_gpu.dW2, HIDDEN_SIZE * S_CODES * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&policy_net_gpu.db2, S_CODES * sizeof(float)));
        CHECK_CUDA(cudaMemset(policy_net_gpu.dW1, 0, S_CODES * HIDDEN_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMemset(policy_net_gpu.db1, 0, HIDDEN_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMemset(policy_net_gpu.dW2, 0, HIDDEN_SIZE * S_CODES * sizeof(float)));
        CHECK_CUDA(cudaMemset(policy_net_gpu.db2, 0, S_CODES * sizeof(float)));

        // Allocate memory for Target Network (weights only)
        CHECK_CUDA(cudaMalloc(&target_net_gpu.W1, S_CODES * HIDDEN_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&target_net_gpu.b1, HIDDEN_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&target_net_gpu.W2, HIDDEN_SIZE * S_CODES * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&target_net_gpu.b2, S_CODES * sizeof(float)));

        // Allocate batch processing buffers
        CHECK_CUDA(cudaMalloc(&policy_net_gpu.d_states, BATCH_SIZE * (S_CODES + 1) * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&policy_net_gpu.d_next_states, BATCH_SIZE * (S_CODES + 1) * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&policy_net_gpu.d_actions, BATCH_SIZE * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&policy_net_gpu.d_rewards, BATCH_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&policy_net_gpu.d_dones, BATCH_SIZE * sizeof(bool)));

        // Allocate intermediate training buffers
        CHECK_CUDA(cudaMalloc(&policy_net_gpu.d_pre_activation_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&policy_net_gpu.d_hidden_activations, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&policy_net_gpu.d_q_values, BATCH_SIZE * S_CODES * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&policy_net_gpu.d_next_q_values, BATCH_SIZE * S_CODES * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&policy_net_gpu.d_delta_out, BATCH_SIZE * S_CODES * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&policy_net_gpu.d_delta_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));

        update_target_network_gpu();
    }

    void free_gpu_memory() {
        // Policy Net
        cudaFree(policy_net_gpu.W1); cudaFree(policy_net_gpu.b1);
        cudaFree(policy_net_gpu.W2); cudaFree(policy_net_gpu.b2);
        cudaFree(policy_net_gpu.dW1); cudaFree(policy_net_gpu.db1);
        cudaFree(policy_net_gpu.dW2); cudaFree(policy_net_gpu.db2);
        // Target Net
        cudaFree(target_net_gpu.W1); cudaFree(target_net_gpu.b1);
        cudaFree(target_net_gpu.W2); cudaFree(target_net_gpu.b2);
        // Batch buffers
        cudaFree(policy_net_gpu.d_states); cudaFree(policy_net_gpu.d_next_states);
        cudaFree(policy_net_gpu.d_actions); cudaFree(policy_net_gpu.d_rewards);
        cudaFree(policy_net_gpu.d_dones);
        // Intermediate buffers
        cudaFree(policy_net_gpu.d_pre_activation_hidden); cudaFree(policy_net_gpu.d_hidden_activations);
        cudaFree(policy_net_gpu.d_q_values); cudaFree(policy_net_gpu.d_next_q_values);
        cudaFree(policy_net_gpu.d_delta_out); cudaFree(policy_net_gpu.d_delta_hidden);
    }

    void update_target_network_gpu() {
        CHECK_CUDA(cudaMemcpy(target_net_gpu.W1, policy_net_gpu.W1, S_CODES * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(target_net_gpu.b1, policy_net_gpu.b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(target_net_gpu.W2, policy_net_gpu.W2, HIDDEN_SIZE * S_CODES * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(target_net_gpu.b2, policy_net_gpu.b2, S_CODES * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    int select_action(const std::vector<float>& state) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        if (dist(rand_gen) < epsilon) {
            std::vector<int> valid_actions;
            for (int i = 0; i < S_CODES; ++i) {
                if (state[i + 1] > 0.5f) valid_actions.push_back(i);
            }
            if (valid_actions.empty()) return std::uniform_int_distribution<int>(0, S_CODES - 1)(rand_gen);
            return valid_actions[std::uniform_int_distribution<int>(0, valid_actions.size() - 1)(rand_gen)];
        }
        else {
            // Copy state to GPU
            CHECK_CUDA(cudaMemcpy(policy_net_gpu.d_states, state.data(), (S_CODES + 1) * sizeof(float), cudaMemcpyHostToDevice));

            // GPU Forward Pass for BATCH_SIZE = 1
            dim3 blockDim(32, 32);
            dim3 gridDim_L1(ceilf(HIDDEN_SIZE / 32.0f), 1);
            forward_kernel << <gridDim_L1, blockDim >> > (policy_net_gpu.d_hidden_activations, nullptr, policy_net_gpu.d_states + 1, policy_net_gpu.W1, policy_net_gpu.b1, 1, S_CODES, HIDDEN_SIZE, true);

            dim3 gridDim_L2(ceilf(S_CODES / 32.0f), 1);
            forward_kernel << <gridDim_L2, blockDim >> > (policy_net_gpu.d_q_values, nullptr, policy_net_gpu.d_hidden_activations, policy_net_gpu.W2, policy_net_gpu.b2, 1, HIDDEN_SIZE, S_CODES, false);

            // Copy Q-values back to CPU
            std::vector<float> q_values(S_CODES);
            CHECK_CUDA(cudaMemcpy(q_values.data(), policy_net_gpu.d_q_values, S_CODES * sizeof(float), cudaMemcpyDeviceToHost));

            // Action Masking on CPU
            for (int i = 0; i < S_CODES; ++i) {
                if (state[i + 1] < 0.5f) { // state is 1-indexed for codes
                    q_values[i] = -std::numeric_limits<float>::infinity();
                }
            }

            return std::distance(q_values.begin(), std::max_element(q_values.begin(), q_values.end()));
        }
    }

    void train() {
        if (memory.size() < BATCH_SIZE) return;
        auto batch = memory.sample(BATCH_SIZE);

        // Prepare batch data on host
        std::vector<float> states_batch(BATCH_SIZE * (S_CODES + 1));
        std::vector<float> next_states_batch(BATCH_SIZE * (S_CODES + 1));
        std::vector<int> actions_batch(BATCH_SIZE);
        std::vector<float> rewards_batch(BATCH_SIZE);

        // FIX #2: Use std::vector<char> for dones because vector<bool> is not contiguous
        std::vector<char> dones_batch(BATCH_SIZE);

        for (int i = 0; i < BATCH_SIZE; ++i) {
            std::copy(batch[i].state.begin(), batch[i].state.end(), states_batch.begin() + i * (S_CODES + 1));
            std::copy(batch[i].next_state.begin(), batch[i].next_state.end(), next_states_batch.begin() + i * (S_CODES + 1));
            actions_batch[i] = batch[i].action;
            rewards_batch[i] = batch[i].reward;
            dones_batch[i] = batch[i].done ? 1 : 0; // Convert bool to 1 or 0 for char vector
        }

        // Copy batch to GPU
        CHECK_CUDA(cudaMemcpy(policy_net_gpu.d_states, states_batch.data(), states_batch.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(policy_net_gpu.d_next_states, next_states_batch.data(), next_states_batch.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(policy_net_gpu.d_actions, actions_batch.data(), actions_batch.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(policy_net_gpu.d_rewards, rewards_batch.data(), rewards_batch.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(policy_net_gpu.d_dones, dones_batch.data(), dones_batch.size() * sizeof(char), cudaMemcpyHostToDevice)); // Note: sizeof(char)

        // --- GPU TRAINING PIPELINE ---

        dim3 blockDim(16, 16);

        // 1. Forward pass on NEXT states with TARGET network to get max_q_next
        dim3 gridDim_target_L1(ceilf(HIDDEN_SIZE / 16.0f), ceilf(BATCH_SIZE / 16.0f));
        forward_kernel << <gridDim_target_L1, blockDim >> > (policy_net_gpu.d_hidden_activations, nullptr, policy_net_gpu.d_next_states + 1, target_net_gpu.W1, target_net_gpu.b1, BATCH_SIZE, S_CODES, HIDDEN_SIZE, true);
        dim3 gridDim_target_L2(ceilf(S_CODES / 16.0f), ceilf(BATCH_SIZE / 16.0f));
        forward_kernel << <gridDim_target_L2, blockDim >> > (policy_net_gpu.d_next_q_values, nullptr, policy_net_gpu.d_hidden_activations, target_net_gpu.W2, target_net_gpu.b2, BATCH_SIZE, HIDDEN_SIZE, S_CODES, false);

        // 2. Forward pass on CURRENT states with POLICY network
        dim3 gridDim_policy_L1(ceilf(HIDDEN_SIZE / 16.0f), ceilf(BATCH_SIZE / 16.0f));
        forward_kernel << <gridDim_policy_L1, blockDim >> > (policy_net_gpu.d_hidden_activations, policy_net_gpu.d_pre_activation_hidden, policy_net_gpu.d_states + 1, policy_net_gpu.W1, policy_net_gpu.b1, BATCH_SIZE, S_CODES, HIDDEN_SIZE, true);
        dim3 gridDim_policy_L2(ceilf(S_CODES / 16.0f), ceilf(BATCH_SIZE / 16.0f));
        forward_kernel << <gridDim_policy_L2, blockDim >> > (policy_net_gpu.d_q_values, nullptr, policy_net_gpu.d_hidden_activations, policy_net_gpu.W2, policy_net_gpu.b2, BATCH_SIZE, HIDDEN_SIZE, S_CODES, false);

        // 3. Calculate TD target and loss delta (initial gradient for backprop)
        calculate_targets_and_loss_delta_kernel << <(BATCH_SIZE + 255) / 256, 256 >> > (policy_net_gpu.d_delta_out, policy_net_gpu.d_q_values, policy_net_gpu.d_next_q_values, policy_net_gpu.d_actions, policy_net_gpu.d_rewards, policy_net_gpu.d_dones, GAMMA, BATCH_SIZE, S_CODES);

        // 4. Backward Pass: Layer 2 (Output -> Hidden)
        dim3 grid_back_L2(ceilf(HIDDEN_SIZE / 16.0f), ceilf(BATCH_SIZE / 16.0f));
        backward_output_layer_kernel << <grid_back_L2, blockDim, 0 >> > (policy_net_gpu.dW2, policy_net_gpu.db2, policy_net_gpu.d_delta_hidden, policy_net_gpu.d_delta_out, policy_net_gpu.W2, policy_net_gpu.d_hidden_activations, policy_net_gpu.d_pre_activation_hidden, BATCH_SIZE, HIDDEN_SIZE, S_CODES);

        // 5. Backward Pass: Layer 1 (Hidden -> Input)
        dim3 grid_back_L1(ceilf(HIDDEN_SIZE / 16.0f), ceilf(S_CODES / 16.0f));
        backward_hidden_layer_kernel << <grid_back_L1, blockDim >> > (policy_net_gpu.dW1, policy_net_gpu.db1, policy_net_gpu.d_delta_hidden, policy_net_gpu.d_states + 1, BATCH_SIZE, S_CODES, HIDDEN_SIZE);

        // 6. Update Weights using SGD
        sgd_update_kernel << <(S_CODES * HIDDEN_SIZE + 255) / 256, 256 >> > (policy_net_gpu.W1, policy_net_gpu.dW1, LEARNING_RATE, S_CODES * HIDDEN_SIZE, BATCH_SIZE);
        sgd_update_kernel << <(HIDDEN_SIZE + 255) / 256, 256 >> > (policy_net_gpu.b1, policy_net_gpu.db1, LEARNING_RATE, HIDDEN_SIZE, BATCH_SIZE);
        sgd_update_kernel << <(HIDDEN_SIZE * S_CODES + 255) / 256, 256 >> > (policy_net_gpu.W2, policy_net_gpu.dW2, LEARNING_RATE, HIDDEN_SIZE * S_CODES, BATCH_SIZE);
        sgd_update_kernel << <(S_CODES + 255) / 256, 256 >> > (policy_net_gpu.b2, policy_net_gpu.db2, LEARNING_RATE, S_CODES, BATCH_SIZE);

        CHECK_CUDA(cudaDeviceSynchronize());
    }

    void update_epsilon() {
        epsilon = std::max(EPSILON_END, epsilon * EPSILON_DECAY);
    }
};

// ===================================================================================
// Main Training Loop
// ===================================================================================
int main() {
    std::cout << "--- Mastermind Deep Q-Network (DQN) Agent - GPU Accelerated ---\n";
    std::cout << "Initializing game data...\n";
    FillSetIterative();
    GenerateMarkTable();

    unsigned int seed = std::random_device{}();
    DQNAgent agent(seed);
    MastermindEnv env(seed);

    std::cout << "Initializing GPU memory...\n";
    agent.init_gpu_memory();

    std::cout << "\n--- Starting Training ---\n\n";

    float total_moves_avg = 15.0f;
    for (int episode = 1; episode <= NUM_EPISODES; ++episode) {
        env.reset();
        int moves = 0;

        for (int t = 0; t < 15; ++t) {
            moves++;
            auto state_copy = env.state;
            int action = agent.select_action(state_copy); // Action is 0-indexed

            float reward;
            bool done = env.step(action + 1, reward); // Env expects 1-indexed

            agent.memory.push({ state_copy, action, reward, env.state, done });
            agent.train();

            if (done) break;
        }

        agent.update_epsilon();

        if (episode % TARGET_UPDATE_FREQ == 0) {
            agent.update_target_network_gpu();
        }

        total_moves_avg = 0.99f * total_moves_avg + 0.01f * moves;
        if (episode % 100 == 0) {
            std::cout << "Episode " << std::setw(5) << episode
                << " | Avg Moves: " << std::fixed << std::setprecision(2) << total_moves_avg
                << " | Epsilon: " << std::fixed << std::setprecision(3) << agent.epsilon << std::endl;
        }
    }

    std::cout << "\nTraining finished.\n";

    std::cout << "\n--- Evaluating trained agent ---\n";
    agent.epsilon = 0.0f; // No exploration during evaluation
    const int EVAL_GAMES = 100;
    int total_eval_moves = 0;

    for (int i = 0; i < EVAL_GAMES; ++i) {
        env.reset();
        int moves = 0;
        for (int t = 0; t < 15; ++t) {
            moves++;
            int action = agent.select_action(env.state);
            float reward;
            bool done = env.step(action + 1, reward);
            if (done) break;
        }
        total_eval_moves += moves;
    }
    std::cout << "\nAverage moves over " << EVAL_GAMES << " games: " << (float)total_eval_moves / EVAL_GAMES << std::endl;

    agent.free_gpu_memory();
    std::cout << "GPU memory freed.\n";
    return 0;
}