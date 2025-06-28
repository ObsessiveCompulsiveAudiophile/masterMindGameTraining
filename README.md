# masterMindGameTraining
A high-performing (GPU accelerated) Mastermind game (4 pegs of 6 repeatable colors) solution agent using Deep Q-Learning

Scores better than Knuth's "Worst case" algorithm (4.47608 guesses average) in a matter of minutes.


How to Compile (using NVIDIA's nvcc):
nvcc -o -O3 -arch=sm_75 --std=c++17 -Xcompiler "/O2"
(Adjust -arch=sm_XX to match your GPU's compute capability, e.g., sm_86 for RTX 30xx)

