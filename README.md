# masterMindGameTraining
A high-performing Mastermind game (4 pegs of 6 repeatable colors) solution agent using Deep Q-Learning


How to Compile (using NVIDIA's nvcc):
nvcc -o Mastermind_DQN_GPU.exe kernel.cu -O3 -arch=sm_75 --std=c++17 -Xcompiler "/O2"
(Adjust -arch=sm_XX to match your GPU's compute capability, e.g., sm_86 for RTX 30xx)

