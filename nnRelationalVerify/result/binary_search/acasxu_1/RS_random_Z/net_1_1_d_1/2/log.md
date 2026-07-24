## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_1.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 1.0495482984


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489)
1: (-0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965)
2: (-0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164)
3: (-0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897)
4: (-0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601)

## BASE Result
execution time: IAR + LP analysis = 1.53 + 1.06 = 2.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1.0564887, upper bound: 1.0564887


# Binary Search by BASE starts (time budget: 1197.42 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1000000, mid=0.1000000, abs_max=1.1883488893508911
rel_dist={0: [-1.0558835892060525, 1.0558835892060525]}

## Binary search (step 1) starts
Candidate diff: 0.0500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0500000, mid=0.0500000, abs_max=1.1883488893508911
rel_dist={0: [-1.0553818619849304, 1.0553818619849311]}

## Binary search (step 2) starts
Candidate diff: 0.0250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0250000, mid=0.0250000, abs_max=1.1883488893508911
rel_dist={0: [-1.0551075589159629, 1.0551075589159629]}

## Binary search (step 3) starts
Candidate diff: 0.0125000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0125000, mid=0.0125000, abs_max=1.1883488893508911
rel_dist={0: [-1.0549470781280466, 1.0549470781280457]}

## Binary search (step 4) starts
Candidate diff: 0.0062500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0062500, mid=0.0062500, abs_max=1.1883488893508911
rel_dist={0: [-1.0548482094125333, 1.0548482094125333]}

## Binary search (step 5) starts
Candidate diff: 0.0031250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0031250, mid=0.0031250, abs_max=1.1883488893508911
rel_dist={0: [-1.0547849699636636, 1.0547849699636629]}

## Binary search (step 6) starts
Candidate diff: 0.0015625


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0015625, mid=0.0015625, abs_max=1.1883488893508911
rel_dist={0: [-1.0547487716980937, 1.054748771698094]}

## Binary search (step 7) starts
Candidate diff: 0.0007812


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0007812, mid=0.0007812, abs_max=1.1883488893508911
rel_dist={0: [-1.054717313123089, 1.0547173131230885]}

## Binary search (step 8) starts
Candidate diff: 0.0003906


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0003906, mid=0.0003906, abs_max=1.1883488893508911
rel_dist={0: [-1.054693067218044, 1.0546930672156352]}

## Binary search (step 9) starts
Candidate diff: 0.0001953


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0001953, mid=0.0001953, abs_max=1.1883488893508911
rel_dist={0: [-1.0546768688586334, 1.0546768688579005]}

## Binary search (step 10) starts
Candidate diff: 0.0000977


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0000977, mid=0.0000977, abs_max=1.1883488893508911
rel_dist={0: [-1.0546645489281865, 1.0546645489275417]}

## Binary search (step 11) starts
Candidate diff: 0.0000488


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000488, mid=0.0000488, abs_max=1.1883488893508911
rel_dist={0: [-1.0546556231562683, 1.0546556237213354]}

## Binary search (step 12) starts
Candidate diff: 0.0000244


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000244, mid=0.0000244, abs_max=1.1883488893508911
rel_dist={0: [-1.0546509986945563, 1.0546509989519355]}

## Binary search (step 13) starts
Candidate diff: 0.0000122


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000122, mid=0.0000122, abs_max=1.1883488893508911
rel_dist={0: [-1.0546486318282617, 1.0546486319569248]}

## Binary search (step 14) starts
Candidate diff: 0.0000061


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000061, mid=0.0000061, abs_max=1.1883488893508911
rel_dist={0: [-1.0546474519922995, 1.054647448669685]}

## Binary search (step 15) starts
Candidate diff: 0.0000031


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000031, mid=0.0000031, abs_max=1.1883488893508911
rel_dist={0: [-1.054646874919929, 1.0546468574895504]}

## Binary search (step 16) starts
Candidate diff: 0.0000015


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000015, mid=0.0000015, abs_max=1.1883488893508911
rel_dist={0: [-1.0546465944949381, 1.0546465655883437]}

## Binary search (step 17) starts
Candidate diff: 0.0000008


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000008, mid=0.0000008, abs_max=1.1883488893508911
rel_dist={0: [-1.05464671617786, 1.0546464864943261]}

## Binary Search Result
Binary search time: 45.93 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1151.48 seconds

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0538449, upper bound: 1.0538449
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0538449, upper bound: 1.0538449
time: 0.37 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.77 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -1.0538449, upper bound: 1.0538449
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -1.0538449, upper bound: 1.0538449

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0481199, upper bound: 1.0481199
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0481199, upper bound: 1.0481199
time: 0.33 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0472786, upper bound: 1.0472786
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0472786, upper bound: 1.0472786
time: 0.33 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.72 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.72
Output dim: 0, lower bound: -1.0481199, upper bound: 1.0481199
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.72
Output dim: 0, lower bound: -1.0481199, upper bound: 1.0481199
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.72
Output dim: 0, lower bound: -1.0472786, upper bound: 1.0472786
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.72
Output dim: 0, lower bound: -1.0472786, upper bound: 1.0472786
Binary search (step 0): status=Status.VERIFIED, low=0.1000000, high=0.2000000, mid=0.1000000, abs_max=1.1883488893508911
rel_dist={0: [-1.0558835892060525, 1.0558835892060525]}

## Binary search (step 1) starts
Candidate diff: 0.1500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0540170, upper bound: 1.0539857
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170
time: 0.36 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.75 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -1.0540170, upper bound: 1.0539857
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0540170, upper bound: 1.0539330
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0540170, upper bound: 1.0539330
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0380455, upper bound: 1.0380455
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0380455, upper bound: 1.0380455
time: 0.33 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.43 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -1.0540170, upper bound: 1.0539330
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -1.0540170, upper bound: 1.0539330
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.43
Output dim: 0, lower bound: -1.0380455, upper bound: 1.0380455
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.43
Output dim: 0, lower bound: -1.0380455, upper bound: 1.0380455

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0380455, upper bound: 1.0380455
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0380455, upper bound: 1.0380455
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0537272, upper bound: 1.0536171
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536263, upper bound: 1.0536249
time: 0.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.10 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.10
Output dim: 0, lower bound: -1.0380455, upper bound: 1.0380455
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.10
Output dim: 0, lower bound: -1.0380455, upper bound: 1.0380455
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 0, lower bound: -1.0537272, upper bound: 1.0536171
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 0, lower bound: -1.0536263, upper bound: 1.0536249

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374981, upper bound: 1.0374981
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374981, upper bound: 1.0374981
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374981, upper bound: 1.0374981
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374981, upper bound: 1.0374981
time: 0.32 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.10 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.10
Output dim: 0, lower bound: -1.0374981, upper bound: 1.0374981
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.10
Output dim: 0, lower bound: -1.0374981, upper bound: 1.0374981
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.10
Output dim: 0, lower bound: -1.0374981, upper bound: 1.0374981
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.10
Output dim: 0, lower bound: -1.0374981, upper bound: 1.0374981
Binary search (step 1): status=Status.VERIFIED, low=0.1500000, high=0.2000000, mid=0.1500000, abs_max=1.1883488893508911
rel_dist={0: [-1.0562380918366323, 1.056238091836632]}

## Binary search (step 2) starts
Candidate diff: 0.1750000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523240, upper bound: 1.0523240
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523240, upper bound: 1.0523240
time: 0.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.67 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -1.0523240, upper bound: 1.0523240
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -1.0523240, upper bound: 1.0523240

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507030, upper bound: 1.0507030
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507030, upper bound: 1.0507030
time: 0.34 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513200, upper bound: 1.0513200
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513200, upper bound: 1.0513200
time: 0.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.08 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.08
Output dim: 0, lower bound: -1.0507030, upper bound: 1.0507030
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.08
Output dim: 0, lower bound: -1.0507030, upper bound: 1.0507030
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.08
Output dim: 0, lower bound: -1.0513200, upper bound: 1.0513200
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.08
Output dim: 0, lower bound: -1.0513200, upper bound: 1.0513200

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475233, upper bound: 1.0475233
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475233, upper bound: 1.0475233
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506845, upper bound: 1.0506811
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506811, upper bound: 1.0506845
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0379136, upper bound: 1.0379136
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0379136, upper bound: 1.0379136
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0511147, upper bound: 1.0510914
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510914, upper bound: 1.0511147
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.24
Output dim: 0, lower bound: -1.0475233, upper bound: 1.0475233
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.24
Output dim: 0, lower bound: -1.0475233, upper bound: 1.0475233
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 0, lower bound: -1.0506845, upper bound: 1.0506811
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 0, lower bound: -1.0506811, upper bound: 1.0506845
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.24
Output dim: 0, lower bound: -1.0379136, upper bound: 1.0379136
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.24
Output dim: 0, lower bound: -1.0379136, upper bound: 1.0379136
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 0, lower bound: -1.0511147, upper bound: 1.0510914
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 0, lower bound: -1.0510914, upper bound: 1.0511147

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506845, upper bound: 1.0506349
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505930, upper bound: 1.0506811
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474200, upper bound: 1.0474103
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475149, upper bound: 1.0474103
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506368, upper bound: 1.0507166
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506368, upper bound: 1.0505489
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0470636, upper bound: 1.0470795
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0470636, upper bound: 1.0470795
time: 0.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.17 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.0506845, upper bound: 1.0506349
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.0505930, upper bound: 1.0506811
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.0474200, upper bound: 1.0474103
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.0475149, upper bound: 1.0474103
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.0506368, upper bound: 1.0507166
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.0506368, upper bound: 1.0505489
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.0470636, upper bound: 1.0470795
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.0470636, upper bound: 1.0470795

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0486685, upper bound: 1.0485830
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0486105, upper bound: 1.0485830
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482300, upper bound: 1.0486200
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0485339, upper bound: 1.0486200
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504232, upper bound: 1.0505395
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507310, upper bound: 1.0504139
time: 0.34 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.25 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.0486685, upper bound: 1.0485830
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.0486105, upper bound: 1.0485830
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.0482300, upper bound: 1.0486200
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.0485339, upper bound: 1.0486200
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.0504232, upper bound: 1.0505395
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.0507310, upper bound: 1.0504139

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506071, upper bound: 1.0504709
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505306, upper bound: 1.0505395
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0465500, upper bound: 1.0464096
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0465234, upper bound: 1.0464096
time: 0.35 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.21 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -1.0506071, upper bound: 1.0504709
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.21
Output dim: 0, lower bound: -1.0505306, upper bound: 1.0505395
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.21
Output dim: 0, lower bound: -1.0465500, upper bound: 1.0464096
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.21
Output dim: 0, lower bound: -1.0465234, upper bound: 1.0464096

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0465234, upper bound: 1.0463616
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0465234, upper bound: 1.0463616
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.32 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.11 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.11
Output dim: 0, lower bound: -1.0465234, upper bound: 1.0463616
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.11
Output dim: 0, lower bound: -1.0465234, upper bound: 1.0463616
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.11
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.11
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
Binary search (step 2): status=Status.VERIFIED, low=0.1750000, high=0.2000000, mid=0.1750000, abs_max=1.1883488893508911
rel_dist={0: [-1.0563961018081798, 1.0563961018081796]}

## Binary search (step 3) starts
Candidate diff: 0.1875000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.70 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.70
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.70
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506381, upper bound: 1.0506381
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506381, upper bound: 1.0506499
time: 0.34 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506381, upper bound: 1.0506381
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506381, upper bound: 1.0506499
time: 0.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.28 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -1.0506381, upper bound: 1.0506381
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -1.0506381, upper bound: 1.0506499
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -1.0506381, upper bound: 1.0506381
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -1.0506381, upper bound: 1.0506499

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506162
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505830
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506381, upper bound: 1.0506371
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506368, upper bound: 1.0506499
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475327, upper bound: 1.0475082
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475327, upper bound: 1.0475082
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505868, upper bound: 1.0506249
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505868, upper bound: 1.0506249
time: 0.35 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.19 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506162
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505830
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.0506381, upper bound: 1.0506371
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.0506368, upper bound: 1.0506499
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.0475327, upper bound: 1.0475082
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.0475327, upper bound: 1.0475082
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.0505868, upper bound: 1.0506249
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.0505868, upper bound: 1.0506249

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505232, upper bound: 1.0505189
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504719
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506057, upper bound: 1.0505689
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506057, upper bound: 1.0505689
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506149
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474970, upper bound: 1.0475327
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474970, upper bound: 1.0475327
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0496788, upper bound: 1.0505739
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0496788, upper bound: 1.0506249
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0470636, upper bound: 1.0470795
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0470636, upper bound: 1.0470795
time: 0.37 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 0, lower bound: -1.0505232, upper bound: 1.0505189
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504719
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 0, lower bound: -1.0506057, upper bound: 1.0505689
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 0, lower bound: -1.0506057, upper bound: 1.0505689
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506149
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.23
Output dim: 0, lower bound: -1.0474970, upper bound: 1.0475327
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.23
Output dim: 0, lower bound: -1.0474970, upper bound: 1.0475327
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 0, lower bound: -1.0496788, upper bound: 1.0505739
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 0, lower bound: -1.0496788, upper bound: 1.0506249
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.23
Output dim: 0, lower bound: -1.0470636, upper bound: 1.0470795
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.23
Output dim: 0, lower bound: -1.0470636, upper bound: 1.0470795

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0480451
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504902, upper bound: 1.0504699
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504902, upper bound: 1.0505689
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0470757, upper bound: 1.0467049
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0470757, upper bound: 1.0467273
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0473495, upper bound: 1.0474396
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0472994, upper bound: 1.0474319
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494887, upper bound: 1.0505412
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505669, upper bound: 1.0505002
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0470636, upper bound: 1.0469022
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0470636, upper bound: 1.0468674
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0493497, upper bound: 1.0505316
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0495176, upper bound: 1.0505268
time: 0.35 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0480451
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0504902, upper bound: 1.0504699
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0504902, upper bound: 1.0505689
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0470757, upper bound: 1.0467049
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0470757, upper bound: 1.0467273
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0473495, upper bound: 1.0474396
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0472994, upper bound: 1.0474319
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0494887, upper bound: 1.0505412
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0505669, upper bound: 1.0505002
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0470636, upper bound: 1.0469022
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0470636, upper bound: 1.0468674
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0493497, upper bound: 1.0505316
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0495176, upper bound: 1.0505268

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505065, upper bound: 1.0503930
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505113, upper bound: 1.0500173
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0473751, upper bound: 1.0473751
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482653, upper bound: 1.0482637
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0483140, upper bound: 1.0483398
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482836, upper bound: 1.0482853
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0483570, upper bound: 1.0483217
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0483553, upper bound: 1.0479569
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0470377, upper bound: 1.0481024
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0470377, upper bound: 1.0481024
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374935, upper bound: 1.0374935
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374935, upper bound: 1.0374935
time: 0.35 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.25 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.0505065, upper bound: 1.0503930
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.0505113, upper bound: 1.0500173
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.0473751, upper bound: 1.0473751
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.0482653, upper bound: 1.0482637
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.0483140, upper bound: 1.0483398
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.0482836, upper bound: 1.0482853
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.0483570, upper bound: 1.0483217
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.0483553, upper bound: 1.0479569
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.0470377, upper bound: 1.0481024
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.0470377, upper bound: 1.0481024
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.0374935, upper bound: 1.0374935
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.0374935, upper bound: 1.0374935

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0476019
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.36 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.21 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0476019
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
Binary search (step 3): status=Status.VERIFIED, low=0.1875000, high=0.2000000, mid=0.1875000, abs_max=1.1883488893508911
rel_dist={0: [-1.0564436068034628, 1.0564436068034624]}

## Binary search (step 4) starts
Candidate diff: 0.1937500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0561869, upper bound: 1.0561869
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0561869, upper bound: 1.0562855
time: 0.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.67 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -1.0561869, upper bound: 1.0561869
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -1.0561869, upper bound: 1.0562855

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
time: 0.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
time: 0.33 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.13 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.13
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.13
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.13
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.13
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506845, upper bound: 1.0506811
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506811, upper bound: 1.0506811
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506121, upper bound: 1.0506154
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506131, upper bound: 1.0506102
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0506509
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505830
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0506223
time: 0.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.13 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.0506845, upper bound: 1.0506811
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.0506811, upper bound: 1.0506811
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.0506121, upper bound: 1.0506154
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.0506131, upper bound: 1.0506102
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0506509
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505830
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0506223

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474200, upper bound: 1.0475149
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474200, upper bound: 1.0475149
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474103, upper bound: 1.0475149
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474103, upper bound: 1.0475149
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506102, upper bound: 1.0506123
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505552, upper bound: 1.0506154
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482900, upper bound: 1.0482772
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482882, upper bound: 1.0476845
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478621, upper bound: 1.0477391
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478761, upper bound: 1.0477375
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505838, upper bound: 1.0506131
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506123, upper bound: 1.0506121
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504876
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223
time: 0.39 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.19
Output dim: 0, lower bound: -1.0474200, upper bound: 1.0475149
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.19
Output dim: 0, lower bound: -1.0474200, upper bound: 1.0475149
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.19
Output dim: 0, lower bound: -1.0474103, upper bound: 1.0475149
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.19
Output dim: 0, lower bound: -1.0474103, upper bound: 1.0475149
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.19
Output dim: 0, lower bound: -1.0506102, upper bound: 1.0506123
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.19
Output dim: 0, lower bound: -1.0505552, upper bound: 1.0506154
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.19
Output dim: 0, lower bound: -1.0482900, upper bound: 1.0482772
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.19
Output dim: 0, lower bound: -1.0482882, upper bound: 1.0476845
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.19
Output dim: 0, lower bound: -1.0478621, upper bound: 1.0477391
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.19
Output dim: 0, lower bound: -1.0478761, upper bound: 1.0477375
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.19
Output dim: 0, lower bound: -1.0505838, upper bound: 1.0506131
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.19
Output dim: 0, lower bound: -1.0506123, upper bound: 1.0506121
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.19
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504876
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.19
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.19
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.19
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505853, upper bound: 1.0505278
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505853, upper bound: 1.0493743
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0481804, upper bound: 1.0482823
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482000, upper bound: 1.0482823
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0380338, upper bound: 1.0380338
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0380338, upper bound: 1.0380338
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0380338, upper bound: 1.0380338
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0380338, upper bound: 1.0380338
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505113, upper bound: 1.0500173
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505113, upper bound: 1.0493246
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505669, upper bound: 1.0505412
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505669, upper bound: 1.0505002
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474507, upper bound: 1.0474686
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0473421, upper bound: 1.0474686
time: 0.34 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.22 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0505853, upper bound: 1.0505278
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0505853, upper bound: 1.0493743
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0481804, upper bound: 1.0482823
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0482000, upper bound: 1.0482823
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0380338, upper bound: 1.0380338
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0380338, upper bound: 1.0380338
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0380338, upper bound: 1.0380338
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0380338, upper bound: 1.0380338
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0505113, upper bound: 1.0500173
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0505113, upper bound: 1.0493246
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0505669, upper bound: 1.0505412
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0505669, upper bound: 1.0505002
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0474507, upper bound: 1.0474686
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0473421, upper bound: 1.0474686

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503581, upper bound: 1.0504499
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0493246, upper bound: 1.0504513
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0378970, upper bound: 1.0378970
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0378970, upper bound: 1.0378970
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505113, upper bound: 1.0493246
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504513, upper bound: 1.0493246
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503162, upper bound: 1.0504310
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504729, upper bound: 1.0504345
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0483570, upper bound: 1.0483217
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0483553, upper bound: 1.0479569
time: 0.32 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.14 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 0, lower bound: -1.0503581, upper bound: 1.0504499
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 0, lower bound: -1.0493246, upper bound: 1.0504513
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.14
Output dim: 0, lower bound: -1.0378970, upper bound: 1.0378970
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.14
Output dim: 0, lower bound: -1.0378970, upper bound: 1.0378970
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.14
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.14
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 0, lower bound: -1.0505113, upper bound: 1.0493246
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 0, lower bound: -1.0504513, upper bound: 1.0493246
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 0, lower bound: -1.0503162, upper bound: 1.0504310
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 0, lower bound: -1.0504729, upper bound: 1.0504345
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.14
Output dim: 0, lower bound: -1.0483570, upper bound: 1.0483217
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.14
Output dim: 0, lower bound: -1.0483553, upper bound: 1.0479569

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475772, upper bound: 1.0480451
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.38 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.63 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0475772, upper bound: 1.0480451
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
Binary search (step 4): status=Status.VERIFIED, low=0.1937500, high=0.2000000, mid=0.1937500, abs_max=1.1883488893508911
rel_dist={0: [-1.0564661451396706, 1.0564661451396704]}

## Binary search (step 5) starts
Candidate diff: 0.1968750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.68 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507322
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507322, upper bound: 1.0507647
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506386, upper bound: 1.0506395
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506395, upper bound: 1.0506386
time: 0.33 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.21 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.21
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507322
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.21
Output dim: 0, lower bound: -1.0507322, upper bound: 1.0507647
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.21
Output dim: 0, lower bound: -1.0506386, upper bound: 1.0506395
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.21
Output dim: 0, lower bound: -1.0506395, upper bound: 1.0506386

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506371, upper bound: 1.0506368
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506381, upper bound: 1.0506371
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0486138, upper bound: 1.0487099
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0486138, upper bound: 1.0487099
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0483561, upper bound: 1.0483151
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0483151, upper bound: 1.0483193
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506395, upper bound: 1.0506086
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506368, upper bound: 1.0506386
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.50 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -1.0506371, upper bound: 1.0506368
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -1.0506381, upper bound: 1.0506371
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.50
Output dim: 0, lower bound: -1.0486138, upper bound: 1.0487099
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.50
Output dim: 0, lower bound: -1.0486138, upper bound: 1.0487099
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.50
Output dim: 0, lower bound: -1.0483561, upper bound: 1.0483151
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.50
Output dim: 0, lower bound: -1.0483151, upper bound: 1.0483193
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -1.0506395, upper bound: 1.0506086
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -1.0506368, upper bound: 1.0506386

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505450, upper bound: 1.0505401
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505462, upper bound: 1.0504961
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0483841, upper bound: 1.0483934
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0483830, upper bound: 1.0483934
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505411, upper bound: 1.0504961
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505411, upper bound: 1.0505152
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498818, upper bound: 1.0506050
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505449, upper bound: 1.0506050
time: 0.37 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.0505450, upper bound: 1.0505401
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.0505462, upper bound: 1.0504961
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.0483841, upper bound: 1.0483934
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.0483830, upper bound: 1.0483934
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.0505411, upper bound: 1.0504961
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.0505411, upper bound: 1.0505152
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.0498818, upper bound: 1.0506050
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.0505449, upper bound: 1.0506050

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505268, upper bound: 1.0504655
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505268, upper bound: 1.0495176
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505316, upper bound: 1.0504064
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505316, upper bound: 1.0493497
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374981, upper bound: 1.0374981
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374981, upper bound: 1.0374981
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480666, upper bound: 1.0480608
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480646, upper bound: 1.0479866
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475686, upper bound: 1.0483030
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0471935, upper bound: 1.0482981
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475686, upper bound: 1.0483434
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482761, upper bound: 1.0483434
time: 0.35 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.50
Output dim: 0, lower bound: -1.0505268, upper bound: 1.0504655
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.50
Output dim: 0, lower bound: -1.0505268, upper bound: 1.0495176
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.50
Output dim: 0, lower bound: -1.0505316, upper bound: 1.0504064
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.50
Output dim: 0, lower bound: -1.0505316, upper bound: 1.0493497
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.50
Output dim: 0, lower bound: -1.0374981, upper bound: 1.0374981
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.50
Output dim: 0, lower bound: -1.0374981, upper bound: 1.0374981
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.50
Output dim: 0, lower bound: -1.0480666, upper bound: 1.0480608
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.50
Output dim: 0, lower bound: -1.0480646, upper bound: 1.0479866
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.50
Output dim: 0, lower bound: -1.0475686, upper bound: 1.0483030
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.50
Output dim: 0, lower bound: -1.0471935, upper bound: 1.0482981
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.50
Output dim: 0, lower bound: -1.0475686, upper bound: 1.0483434
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.50
Output dim: 0, lower bound: -1.0482761, upper bound: 1.0483434

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0476170, upper bound: 1.0480716
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0481269, upper bound: 1.0480716
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480729, upper bound: 1.0470377
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480973, upper bound: 1.0471607
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0481114, upper bound: 1.0480103
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0470377, upper bound: 1.0476156
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505143, upper bound: 1.0493246
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505113, upper bound: 1.0493246
time: 0.37 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 0, lower bound: -1.0476170, upper bound: 1.0480716
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 0, lower bound: -1.0481269, upper bound: 1.0480716
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 0, lower bound: -1.0480729, upper bound: 1.0470377
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 0, lower bound: -1.0480973, upper bound: 1.0471607
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 0, lower bound: -1.0481114, upper bound: 1.0480103
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 0, lower bound: -1.0470377, upper bound: 1.0476156
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -1.0505143, upper bound: 1.0493246
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -1.0505113, upper bound: 1.0493246

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.40 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.23
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.23
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.23
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.23
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
Binary search (step 5): status=Status.VERIFIED, low=0.1968750, high=0.2000000, mid=0.1968750, abs_max=1.1883488893508911
rel_dist={0: [-1.0564774142809072, 1.056477414280907]}

## Binary search (step 6) starts
Candidate diff: 0.1984375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523240, upper bound: 1.0523240
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523240, upper bound: 1.0523240
time: 0.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.66 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -1.0523240, upper bound: 1.0523240
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -1.0523240, upper bound: 1.0523240

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513564, upper bound: 1.0513564
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513564, upper bound: 1.0516066
time: 0.37 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0522159, upper bound: 1.0523240
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0522159, upper bound: 1.0522414
time: 0.37 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.16 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -1.0513564, upper bound: 1.0513564
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -1.0513564, upper bound: 1.0516066
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -1.0522159, upper bound: 1.0523240
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -1.0522159, upper bound: 1.0522414

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513133, upper bound: 1.0512955
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512979, upper bound: 1.0513133
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0379136, upper bound: 1.0379136
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0379136, upper bound: 1.0379136
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506539, upper bound: 1.0506146
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507030, upper bound: 1.0506146
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506539, upper bound: 1.0507030
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506146, upper bound: 1.0507030
time: 0.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.20 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 0, lower bound: -1.0513133, upper bound: 1.0512955
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 0, lower bound: -1.0512979, upper bound: 1.0513133
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.20
Output dim: 0, lower bound: -1.0379136, upper bound: 1.0379136
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.20
Output dim: 0, lower bound: -1.0379136, upper bound: 1.0379136
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 0, lower bound: -1.0506539, upper bound: 1.0506146
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 0, lower bound: -1.0507030, upper bound: 1.0506146
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 0, lower bound: -1.0506539, upper bound: 1.0507030
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 0, lower bound: -1.0506146, upper bound: 1.0507030

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0515415, upper bound: 1.0512682
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512755, upper bound: 1.0512655
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0511706, upper bound: 1.0512755
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0511706, upper bound: 1.0512865
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475233, upper bound: 1.0472821
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475233, upper bound: 1.0472626
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506845, upper bound: 1.0505811
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506811, upper bound: 1.0505930
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504750, upper bound: 1.0506091
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505449, upper bound: 1.0506050
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505739, upper bound: 1.0505868
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505500, upper bound: 1.0506249
time: 0.34 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -1.0515415, upper bound: 1.0512682
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -1.0512755, upper bound: 1.0512655
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -1.0511706, upper bound: 1.0512755
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -1.0511706, upper bound: 1.0512865
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.20
Output dim: 0, lower bound: -1.0475233, upper bound: 1.0472821
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.20
Output dim: 0, lower bound: -1.0475233, upper bound: 1.0472626
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -1.0506845, upper bound: 1.0505811
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -1.0506811, upper bound: 1.0505930
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -1.0504750, upper bound: 1.0506091
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -1.0505449, upper bound: 1.0506050
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -1.0505739, upper bound: 1.0505868
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -1.0505500, upper bound: 1.0506249

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498621, upper bound: 1.0500896
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500803, upper bound: 1.0500845
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498621, upper bound: 1.0500861
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498621, upper bound: 1.0500879
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474103, upper bound: 1.0472788
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474103, upper bound: 1.0472457
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505837, upper bound: 1.0498599
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505869, upper bound: 1.0504529
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505196, upper bound: 1.0505869
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505301, upper bound: 1.0505909
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504674, upper bound: 1.0503377
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504655, upper bound: 1.0505268
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0468674, upper bound: 1.0470636
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469022, upper bound: 1.0470636
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504699, upper bound: 1.0506057
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505345, upper bound: 1.0506077
time: 0.36 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.0498621, upper bound: 1.0500896
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.0500803, upper bound: 1.0500845
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.0498621, upper bound: 1.0500861
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.0498621, upper bound: 1.0500879
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.0474103, upper bound: 1.0472788
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.0474103, upper bound: 1.0472457
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.0505837, upper bound: 1.0498599
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.0505869, upper bound: 1.0504529
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.0505196, upper bound: 1.0505869
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.0505301, upper bound: 1.0505909
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.0504674, upper bound: 1.0503377
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.0504655, upper bound: 1.0505268
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.0468674, upper bound: 1.0470636
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.0469022, upper bound: 1.0470636
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.0504699, upper bound: 1.0506057
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.0505345, upper bound: 1.0506077

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500608, upper bound: 1.0500896
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498621, upper bound: 1.0500535
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480612
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480612
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475752, upper bound: 1.0480612
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475752, upper bound: 1.0480612
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0471584, upper bound: 1.0471584
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482740, upper bound: 1.0475315
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0378970, upper bound: 1.0378970
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0378970, upper bound: 1.0378970
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0481804, upper bound: 1.0482823
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0481804, upper bound: 1.0482823
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0378970, upper bound: 1.0378970
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0378970, upper bound: 1.0378970
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374935, upper bound: 1.0374935
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374935, upper bound: 1.0374935
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374935, upper bound: 1.0374935
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374935, upper bound: 1.0374935
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0467050, upper bound: 1.0470757
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0467049, upper bound: 1.0470757
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0483616, upper bound: 1.0484274
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0473751, upper bound: 1.0484274
time: 0.40 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0500608, upper bound: 1.0500896
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0498621, upper bound: 1.0500535
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480612
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480612
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0475752, upper bound: 1.0480612
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0475752, upper bound: 1.0480612
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0471584, upper bound: 1.0471584
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0482740, upper bound: 1.0475315
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0378970, upper bound: 1.0378970
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0378970, upper bound: 1.0378970
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0481804, upper bound: 1.0482823
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0481804, upper bound: 1.0482823
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0378970, upper bound: 1.0378970
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0378970, upper bound: 1.0378970
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0374935, upper bound: 1.0374935
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0374935, upper bound: 1.0374935
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0374935, upper bound: 1.0374935
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0374935, upper bound: 1.0374935
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0467050, upper bound: 1.0470757
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0467049, upper bound: 1.0470757
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0483616, upper bound: 1.0484274
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.32
Output dim: 0, lower bound: -1.0473751, upper bound: 1.0484274

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.34 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.27 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
Binary search (step 6): status=Status.VERIFIED, low=0.1984375, high=0.2000000, mid=0.1984375, abs_max=1.1883488893508911
rel_dist={0: [-1.0564830488246575, 1.0564830488246573]}

## Binary search (step 7) starts
Candidate diff: 0.1992187


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0540170, upper bound: 1.0539857
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170
time: 0.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.70 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.70
Output dim: 0, lower bound: -1.0540170, upper bound: 1.0539857
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.70
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519933, upper bound: 1.0519933
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519933, upper bound: 1.0519933
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.20 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -1.0519933, upper bound: 1.0519933
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -1.0519933, upper bound: 1.0519933
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0379908, upper bound: 1.0379908
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0379908, upper bound: 1.0379908
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0379908, upper bound: 1.0379908
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0379908, upper bound: 1.0379908
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0537705, upper bound: 1.0538344
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0537639, upper bound: 1.0538548
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0380455, upper bound: 1.0380455
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0380455, upper bound: 1.0380455
time: 0.33 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.23 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.23
Output dim: 0, lower bound: -1.0379908, upper bound: 1.0379908
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.23
Output dim: 0, lower bound: -1.0379908, upper bound: 1.0379908
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.23
Output dim: 0, lower bound: -1.0379908, upper bound: 1.0379908
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.23
Output dim: 0, lower bound: -1.0379908, upper bound: 1.0379908
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.23
Output dim: 0, lower bound: -1.0537705, upper bound: 1.0538344
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.23
Output dim: 0, lower bound: -1.0537639, upper bound: 1.0538548
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.23
Output dim: 0, lower bound: -1.0380455, upper bound: 1.0380455
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.23
Output dim: 0, lower bound: -1.0380455, upper bound: 1.0380455

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513120, upper bound: 1.0515634
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513120, upper bound: 1.0515609
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534525, upper bound: 1.0534652
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534525, upper bound: 1.0535664
time: 0.39 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.28 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -1.0513120, upper bound: 1.0515634
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -1.0513120, upper bound: 1.0515609
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -1.0534525, upper bound: 1.0534652
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -1.0534525, upper bound: 1.0535664

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501733, upper bound: 1.0503376
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501766, upper bound: 1.0501755
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512595, upper bound: 1.0512701
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0511706, upper bound: 1.0515242
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512391, upper bound: 1.0512671
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512595, upper bound: 1.0511706
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516319, upper bound: 1.0517173
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516319, upper bound: 1.0517173
time: 0.37 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.25 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.0501733, upper bound: 1.0503376
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.0501766, upper bound: 1.0501755
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.0512595, upper bound: 1.0512701
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.0511706, upper bound: 1.0515242
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.0512391, upper bound: 1.0512671
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.0512595, upper bound: 1.0511706
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.0516319, upper bound: 1.0517173
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.0516319, upper bound: 1.0517173

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482900, upper bound: 1.0482370
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482900, upper bound: 1.0482370
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482882, upper bound: 1.0476824
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482882, upper bound: 1.0476845
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500700, upper bound: 1.0501010
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500739, upper bound: 1.0500916
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500753, upper bound: 1.0498621
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500752, upper bound: 1.0498621
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374606, upper bound: 1.0374606
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374606, upper bound: 1.0374606
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
time: 0.39 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.39 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.0482900, upper bound: 1.0482370
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.0482900, upper bound: 1.0482370
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.0482882, upper bound: 1.0476824
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.0482882, upper bound: 1.0476845
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.0500700, upper bound: 1.0501010
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.0500739, upper bound: 1.0500916
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.0500753, upper bound: 1.0498621
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.0500752, upper bound: 1.0498621
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.0374606, upper bound: 1.0374606
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.0374606, upper bound: 1.0374606
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480724, upper bound: 1.0469980
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480724, upper bound: 1.0469980
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.34 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.17 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.17
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.17
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.17
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.17
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.17
Output dim: 0, lower bound: -1.0480724, upper bound: 1.0469980
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.17
Output dim: 0, lower bound: -1.0480724, upper bound: 1.0469980
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.17
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.17
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
Binary search (step 7): status=Status.VERIFIED, low=0.1992187, high=0.2000000, mid=0.1992187, abs_max=1.1883488893508911
rel_dist={0: [-1.0564858661234005, 1.0564858661234]}

## Binary search (step 8) starts
Candidate diff: 0.1996094


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523240, upper bound: 1.0523240
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523240, upper bound: 1.0523240
time: 0.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.67 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -1.0523240, upper bound: 1.0523240
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -1.0523240, upper bound: 1.0523240

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523033, upper bound: 1.0522573
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0522543, upper bound: 1.0523033
time: 0.33 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0522159, upper bound: 1.0523240
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0522159, upper bound: 1.0522414
time: 0.37 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.19 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -1.0523033, upper bound: 1.0522573
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -1.0522543, upper bound: 1.0523033
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -1.0522159, upper bound: 1.0523240
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -1.0522159, upper bound: 1.0522414

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520579, upper bound: 1.0522573
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523033, upper bound: 1.0520579
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0515609, upper bound: 1.0513290
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513206, upper bound: 1.0515733
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510958, upper bound: 1.0513464
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510958, upper bound: 1.0510663
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0522922, upper bound: 1.0516768
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517134, upper bound: 1.0522146
time: 0.45 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.31 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -1.0520579, upper bound: 1.0522573
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -1.0523033, upper bound: 1.0520579
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -1.0515609, upper bound: 1.0513290
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -1.0513206, upper bound: 1.0515733
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -1.0510958, upper bound: 1.0513464
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -1.0510958, upper bound: 1.0510663
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -1.0522922, upper bound: 1.0516768
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -1.0517134, upper bound: 1.0522146

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512876, upper bound: 1.0513144
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512876, upper bound: 1.0513144
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510395, upper bound: 1.0511970
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510395, upper bound: 1.0510260
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505837, upper bound: 1.0505909
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505837, upper bound: 1.0505909
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513120, upper bound: 1.0515733
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513206, upper bound: 1.0514933
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508433, upper bound: 1.0509566
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508433, upper bound: 1.0509566
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0486525, upper bound: 1.0482711
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0486525, upper bound: 1.0482711
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513132, upper bound: 1.0512769
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513132, upper bound: 1.0512303
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510810, upper bound: 1.0511147
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510810, upper bound: 1.0511147
time: 0.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.16
Output dim: 0, lower bound: -1.0512876, upper bound: 1.0513144
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.16
Output dim: 0, lower bound: -1.0512876, upper bound: 1.0513144
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.16
Output dim: 0, lower bound: -1.0510395, upper bound: 1.0511970
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.16
Output dim: 0, lower bound: -1.0510395, upper bound: 1.0510260
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.16
Output dim: 0, lower bound: -1.0505837, upper bound: 1.0505909
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.16
Output dim: 0, lower bound: -1.0505837, upper bound: 1.0505909
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.16
Output dim: 0, lower bound: -1.0513120, upper bound: 1.0515733
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.16
Output dim: 0, lower bound: -1.0513206, upper bound: 1.0514933
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.16
Output dim: 0, lower bound: -1.0508433, upper bound: 1.0509566
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.16
Output dim: 0, lower bound: -1.0508433, upper bound: 1.0509566
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.16
Output dim: 0, lower bound: -1.0486525, upper bound: 1.0482711
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.16
Output dim: 0, lower bound: -1.0486525, upper bound: 1.0482711
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.16
Output dim: 0, lower bound: -1.0513132, upper bound: 1.0512769
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.16
Output dim: 0, lower bound: -1.0513132, upper bound: 1.0512303
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.16
Output dim: 0, lower bound: -1.0510810, upper bound: 1.0511147
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.16
Output dim: 0, lower bound: -1.0510810, upper bound: 1.0511147

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508911, upper bound: 1.0509452
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507978, upper bound: 1.0508254
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510098, upper bound: 1.0510759
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506301, upper bound: 1.0510700
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508119, upper bound: 1.0509372
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508159, upper bound: 1.0509372
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0485339, upper bound: 1.0486200
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0485339, upper bound: 1.0486200
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475989, upper bound: 1.0482784
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482772, upper bound: 1.0482784
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0471584, upper bound: 1.0482784
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482772, upper bound: 1.0482784
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505869, upper bound: 1.0505196
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0493743, upper bound: 1.0505196
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0493743, upper bound: 1.0505853
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0493743, upper bound: 1.0505853
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506368, upper bound: 1.0507166
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505489, upper bound: 1.0507195
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506368, upper bound: 1.0507166
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503840, upper bound: 1.0507195
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0515415, upper bound: 1.0512391
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512755, upper bound: 1.0512496
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512865, upper bound: 1.0512015
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512755, upper bound: 1.0511706
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374935, upper bound: 1.0374935
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374935, upper bound: 1.0374935
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469151, upper bound: 1.0470795
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469151, upper bound: 1.0470795
time: 0.37 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0508911, upper bound: 1.0509452
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0507978, upper bound: 1.0508254
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0510098, upper bound: 1.0510759
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0506301, upper bound: 1.0510700
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0508119, upper bound: 1.0509372
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0508159, upper bound: 1.0509372
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0485339, upper bound: 1.0486200
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0485339, upper bound: 1.0486200
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0475989, upper bound: 1.0482784
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0482772, upper bound: 1.0482784
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0471584, upper bound: 1.0482784
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0482772, upper bound: 1.0482784
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0505869, upper bound: 1.0505196
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0493743, upper bound: 1.0505196
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0493743, upper bound: 1.0505853
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0493743, upper bound: 1.0505853
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0506368, upper bound: 1.0507166
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0505489, upper bound: 1.0507195
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0506368, upper bound: 1.0507166
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0503840, upper bound: 1.0507195
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0515415, upper bound: 1.0512391
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0512755, upper bound: 1.0512496
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0512865, upper bound: 1.0512015
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0512755, upper bound: 1.0511706
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0374935, upper bound: 1.0374935
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0374935, upper bound: 1.0374935
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0469151, upper bound: 1.0470795
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0469151, upper bound: 1.0470795

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0376684, upper bound: 1.0376684
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0376684, upper bound: 1.0376684
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469874, upper bound: 1.0469814
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0467554, upper bound: 1.0469814
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501750, upper bound: 1.0507133
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501750, upper bound: 1.0504983
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0376684, upper bound: 1.0376684
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0376684, upper bound: 1.0376684
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504883, upper bound: 1.0507099
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501750, upper bound: 1.0507310
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0378970, upper bound: 1.0378970
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0378970, upper bound: 1.0378970
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482823, upper bound: 1.0482000
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0471584, upper bound: 1.0481804
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0378970, upper bound: 1.0378970
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0378970, upper bound: 1.0378970
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0471584, upper bound: 1.0482757
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0471584, upper bound: 1.0482682
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0465634, upper bound: 1.0463886
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0465634, upper bound: 1.0463886
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500861, upper bound: 1.0500240
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500751, upper bound: 1.0498621
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500820, upper bound: 1.0498621
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500752, upper bound: 1.0498621
time: 0.38 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0376684, upper bound: 1.0376684
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0376684, upper bound: 1.0376684
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0469874, upper bound: 1.0469814
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0467554, upper bound: 1.0469814
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0501750, upper bound: 1.0507133
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0501750, upper bound: 1.0504983
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0376684, upper bound: 1.0376684
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0376684, upper bound: 1.0376684
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0504883, upper bound: 1.0507099
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0501750, upper bound: 1.0507310
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0378970, upper bound: 1.0378970
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0378970, upper bound: 1.0378970
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0482823, upper bound: 1.0482000
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0471584, upper bound: 1.0481804
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0378970, upper bound: 1.0378970
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0378970, upper bound: 1.0378970
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0471584, upper bound: 1.0482757
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0471584, upper bound: 1.0482682
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0465634, upper bound: 1.0463886
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0465634, upper bound: 1.0463886
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0500861, upper bound: 1.0500240
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0500751, upper bound: 1.0498621
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0500820, upper bound: 1.0498621
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -1.0500752, upper bound: 1.0498621

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0463616, upper bound: 1.0464728
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0463616, upper bound: 1.0464728
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0463677, upper bound: 1.0468067
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0463677, upper bound: 1.0468067
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0463616, upper bound: 1.0468271
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0463616, upper bound: 1.0468271
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.33 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0463616, upper bound: 1.0464728
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0463616, upper bound: 1.0464728
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0463677, upper bound: 1.0468067
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0463677, upper bound: 1.0468067
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0463616, upper bound: 1.0468271
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0463616, upper bound: 1.0468271
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
Binary search (step 8): status=Status.VERIFIED, low=0.1996094, high=0.2000000, mid=0.1996094, abs_max=1.1883488893508911
rel_dist={0: [-1.0564872747996392, 1.0564872747996397]}

## Binary search (step 9) starts
Candidate diff: 0.1998047


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0560075
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0544116
time: 0.36 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.74 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.74
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0560075
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.74
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0544116

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0542186, upper bound: 1.0557202
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0542186, upper bound: 1.0558319
time: 0.37 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0544116
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0560075, upper bound: 1.0544116
time: 0.39 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.19 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -1.0542186, upper bound: 1.0557202
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -1.0542186, upper bound: 1.0558319
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0544116
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -1.0560075, upper bound: 1.0544116

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519048, upper bound: 1.0518986
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519048, upper bound: 1.0518986
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0541488, upper bound: 1.0558174
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0542302, upper bound: 1.0541488
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0487099, upper bound: 1.0486138
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0486138, upper bound: 1.0486138
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0540476, upper bound: 1.0540459
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0540558, upper bound: 1.0540693
time: 0.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.27 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 0, lower bound: -1.0519048, upper bound: 1.0518986
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 0, lower bound: -1.0519048, upper bound: 1.0518986
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 0, lower bound: -1.0541488, upper bound: 1.0558174
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 0, lower bound: -1.0542302, upper bound: 1.0541488
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 0, lower bound: -1.0487099, upper bound: 1.0486138
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 0, lower bound: -1.0486138, upper bound: 1.0486138
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 0, lower bound: -1.0540476, upper bound: 1.0540459
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 0, lower bound: -1.0540558, upper bound: 1.0540693

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501656, upper bound: 1.0501841
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502038, upper bound: 1.0501832
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482823, upper bound: 1.0482772
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482823, upper bound: 1.0482772
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0538559, upper bound: 1.0538973
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0538687, upper bound: 1.0555559
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0472131, upper bound: 1.0472604
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0472131, upper bound: 1.0472604
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517943, upper bound: 1.0517251
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517951, upper bound: 1.0517232
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0466274, upper bound: 1.0457457
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0466274, upper bound: 1.0457457
time: 0.34 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 0, lower bound: -1.0501656, upper bound: 1.0501841
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 0, lower bound: -1.0502038, upper bound: 1.0501832
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.10
Output dim: 0, lower bound: -1.0482823, upper bound: 1.0482772
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.10
Output dim: 0, lower bound: -1.0482823, upper bound: 1.0482772
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 0, lower bound: -1.0538559, upper bound: 1.0538973
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 0, lower bound: -1.0538687, upper bound: 1.0555559
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.10
Output dim: 0, lower bound: -1.0472131, upper bound: 1.0472604
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.10
Output dim: 0, lower bound: -1.0472131, upper bound: 1.0472604
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 0, lower bound: -1.0517943, upper bound: 1.0517251
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.10
Output dim: 0, lower bound: -1.0517951, upper bound: 1.0517232
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.10
Output dim: 0, lower bound: -1.0466274, upper bound: 1.0457457
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.10
Output dim: 0, lower bound: -1.0466274, upper bound: 1.0457457

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0376684, upper bound: 1.0376684
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0376684, upper bound: 1.0376684
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0376684, upper bound: 1.0376684
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0376684, upper bound: 1.0376684
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374606, upper bound: 1.0374606
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374606, upper bound: 1.0374606
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505395, upper bound: 1.0506071
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504105, upper bound: 1.0506071
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503260, upper bound: 1.0500846
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501441, upper bound: 1.0500846
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501082, upper bound: 1.0500342
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501082, upper bound: 1.0498913
time: 0.34 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.68 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.68
Output dim: 0, lower bound: -1.0376684, upper bound: 1.0376684
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.68
Output dim: 0, lower bound: -1.0376684, upper bound: 1.0376684
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.68
Output dim: 0, lower bound: -1.0376684, upper bound: 1.0376684
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.68
Output dim: 0, lower bound: -1.0376684, upper bound: 1.0376684
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.68
Output dim: 0, lower bound: -1.0374606, upper bound: 1.0374606
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.68
Output dim: 0, lower bound: -1.0374606, upper bound: 1.0374606
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -1.0505395, upper bound: 1.0506071
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -1.0504105, upper bound: 1.0506071
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -1.0503260, upper bound: 1.0500846
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -1.0501441, upper bound: 1.0500846
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -1.0501082, upper bound: 1.0500342
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.68
Output dim: 0, lower bound: -1.0501082, upper bound: 1.0498913

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505395, upper bound: 1.0505306
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504709, upper bound: 1.0506071
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503015, upper bound: 1.0500534
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498621, upper bound: 1.0500526
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503015, upper bound: 1.0500513
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500763, upper bound: 1.0500488
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.31 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.45 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -1.0505395, upper bound: 1.0505306
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -1.0504709, upper bound: 1.0506071
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.45
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.45
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -1.0503015, upper bound: 1.0500534
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -1.0498621, upper bound: 1.0500526
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -1.0503015, upper bound: 1.0500513
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -1.0500763, upper bound: 1.0500488
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.45
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.45
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.45
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.45
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0465086, upper bound: 1.0464370
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0465086, upper bound: 1.0464370
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0463616, upper bound: 1.0465234
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0463616, upper bound: 1.0465234
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478402, upper bound: 1.0479386
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478402, upper bound: 1.0479386
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.33 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.54 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 0, lower bound: -1.0465086, upper bound: 1.0464370
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 0, lower bound: -1.0465086, upper bound: 1.0464370
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 0, lower bound: -1.0463616, upper bound: 1.0465234
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 0, lower bound: -1.0463616, upper bound: 1.0465234
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 0, lower bound: -1.0478402, upper bound: 1.0479386
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 0, lower bound: -1.0478402, upper bound: 1.0479386
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.54
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
Binary search (step 9): status=Status.VERIFIED, low=0.1998047, high=0.2000000, mid=0.1998047, abs_max=1.1883488893508911
rel_dist={0: [-1.0564879791108912, 1.0564879791108908]}

## Binary search (step 10) starts
Candidate diff: 0.1999023


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0553725, upper bound: 1.0553725
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0553725, upper bound: 1.0562861
time: 0.31 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.77 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -1.0553725, upper bound: 1.0553725
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -1.0553725, upper bound: 1.0562861

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506499, upper bound: 1.0506381
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506381, upper bound: 1.0506381
time: 0.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0472786, upper bound: 1.0472786
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0472786, upper bound: 1.0472786
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.25 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.0506499, upper bound: 1.0506381
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.0506381, upper bound: 1.0506381
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.0472786, upper bound: 1.0472786
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.0472786, upper bound: 1.0472786

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475327, upper bound: 1.0475082
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475327, upper bound: 1.0475082
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475327, upper bound: 1.0475082
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475327, upper bound: 1.0475082
time: 0.32 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.06 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.06
Output dim: 0, lower bound: -1.0475327, upper bound: 1.0475082
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.06
Output dim: 0, lower bound: -1.0475327, upper bound: 1.0475082
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.06
Output dim: 0, lower bound: -1.0475327, upper bound: 1.0475082
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.06
Output dim: 0, lower bound: -1.0475327, upper bound: 1.0475082
Binary search (step 10): status=Status.VERIFIED, low=0.1999023, high=0.2000000, mid=0.1999023, abs_max=1.1883488893508911
rel_dist={0: [-1.0564883312396494, 1.056488331239649]}

## Binary search (step 11) starts
Candidate diff: 0.1999512


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523240, upper bound: 1.0523240
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523240, upper bound: 1.0523240
time: 0.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.70 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.70
Output dim: 0, lower bound: -1.0523240, upper bound: 1.0523240
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.70
Output dim: 0, lower bound: -1.0523240, upper bound: 1.0523240

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510958, upper bound: 1.0513464
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510958, upper bound: 1.0510958
time: 0.34 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507030, upper bound: 1.0507030
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507030, upper bound: 1.0507030
time: 0.33 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.54 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 0, lower bound: -1.0510958, upper bound: 1.0513464
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 0, lower bound: -1.0510958, upper bound: 1.0510958
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 0, lower bound: -1.0507030, upper bound: 1.0507030
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 0, lower bound: -1.0507030, upper bound: 1.0507030

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510958, upper bound: 1.0513464
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510663, upper bound: 1.0513401
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513325, upper bound: 1.0510260
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512592, upper bound: 1.0510728
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506050, upper bound: 1.0506091
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506091, upper bound: 1.0506050
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0486525, upper bound: 1.0486990
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0486525, upper bound: 1.0486990
time: 0.35 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.13 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.0510958, upper bound: 1.0513464
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.0510663, upper bound: 1.0513401
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.0513325, upper bound: 1.0510260
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.0512592, upper bound: 1.0510728
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.0506050, upper bound: 1.0506091
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.0506091, upper bound: 1.0506050
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.0486525, upper bound: 1.0486990
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.0486525, upper bound: 1.0486990

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509396, upper bound: 1.0508944
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508470, upper bound: 1.0512438
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502271, upper bound: 1.0502094
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502115, upper bound: 1.0504004
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501656, upper bound: 1.0501771
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501790, upper bound: 1.0501755
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0486200, upper bound: 1.0486101
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0486243, upper bound: 1.0486101
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0379136, upper bound: 1.0379136
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0379136, upper bound: 1.0379136
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0483052, upper bound: 1.0483434
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0483052, upper bound: 1.0483434
time: 0.32 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.07 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.0509396, upper bound: 1.0508944
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.0508470, upper bound: 1.0512438
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.0502271, upper bound: 1.0502094
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.0502115, upper bound: 1.0504004
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.0501656, upper bound: 1.0501771
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.0501790, upper bound: 1.0501755
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.0486200, upper bound: 1.0486101
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.0486243, upper bound: 1.0486101
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.0379136, upper bound: 1.0379136
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.0379136, upper bound: 1.0379136
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.0483052, upper bound: 1.0483434
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.0483052, upper bound: 1.0483434

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509096, upper bound: 1.0508690
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508537, upper bound: 1.0508314
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498913, upper bound: 1.0501172
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500846, upper bound: 1.0503260
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0377284, upper bound: 1.0377284
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0377284, upper bound: 1.0377284
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0471935, upper bound: 1.0483030
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475686, upper bound: 1.0483030
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0376684, upper bound: 1.0376684
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0376684, upper bound: 1.0376684
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482784, upper bound: 1.0476824
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482784, upper bound: 1.0476845
time: 0.38 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0509096, upper bound: 1.0508690
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0508537, upper bound: 1.0508314
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0498913, upper bound: 1.0501172
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0500846, upper bound: 1.0503260
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0377284, upper bound: 1.0377284
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0377284, upper bound: 1.0377284
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0471935, upper bound: 1.0483030
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0475686, upper bound: 1.0483030
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0376684, upper bound: 1.0376684
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0376684, upper bound: 1.0376684
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0482784, upper bound: 1.0476824
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -1.0482784, upper bound: 1.0476845

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504871, upper bound: 1.0507104
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504883, upper bound: 1.0507104
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500608, upper bound: 1.0500887
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500700, upper bound: 1.0501010
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0470377, upper bound: 1.0480754
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0470377, upper bound: 1.0480754
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480202, upper bound: 1.0480608
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479854, upper bound: 1.0480608
time: 0.35 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.24
Output dim: 0, lower bound: -1.0504871, upper bound: 1.0507104
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.24
Output dim: 0, lower bound: -1.0504883, upper bound: 1.0507104
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.24
Output dim: 0, lower bound: -1.0500608, upper bound: 1.0500887
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.24
Output dim: 0, lower bound: -1.0500700, upper bound: 1.0501010
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.24
Output dim: 0, lower bound: -1.0470377, upper bound: 1.0480754
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.24
Output dim: 0, lower bound: -1.0470377, upper bound: 1.0480754
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.24
Output dim: 0, lower bound: -1.0480202, upper bound: 1.0480608
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.24
Output dim: 0, lower bound: -1.0479854, upper bound: 1.0480608

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0465285, upper bound: 1.0466977
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0465285, upper bound: 1.0466977
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0465285, upper bound: 1.0466977
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0465285, upper bound: 1.0466977
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.34 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.48 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -1.0465285, upper bound: 1.0466977
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -1.0465285, upper bound: 1.0466977
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -1.0465285, upper bound: 1.0466977
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -1.0465285, upper bound: 1.0466977
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
Binary search (step 11): status=Status.VERIFIED, low=0.1999512, high=0.2000000, mid=0.1999512, abs_max=1.1883488893508911
rel_dist={0: [-1.0564885073308965, 1.0564885073308963]}

## Binary search (step 12) starts
Candidate diff: 0.1999756


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0560075
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0544116
time: 0.41 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.77 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0560075
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0544116

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519933, upper bound: 1.0519933
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519933, upper bound: 1.0519933
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0487099, upper bound: 1.0487099
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0487099, upper bound: 1.0487099
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.03 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -1.0519933, upper bound: 1.0519933
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -1.0519933, upper bound: 1.0519933
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.03
Output dim: 0, lower bound: -1.0487099, upper bound: 1.0487099
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.03
Output dim: 0, lower bound: -1.0487099, upper bound: 1.0487099

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519667, upper bound: 1.0519933
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519933, upper bound: 1.0519406
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0379908, upper bound: 1.0379908
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0379908, upper bound: 1.0379908
time: 0.35 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.14 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.0519667, upper bound: 1.0519933
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.0519933, upper bound: 1.0519406
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.0379908, upper bound: 1.0379908
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.0379908, upper bound: 1.0379908

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502376, upper bound: 1.0502167
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502103, upper bound: 1.0502167
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482102, upper bound: 1.0483151
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482102, upper bound: 1.0483151
time: 0.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0502376, upper bound: 1.0502167
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0502103, upper bound: 1.0502167
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0482102, upper bound: 1.0483151
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0482102, upper bound: 1.0483151

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482981, upper bound: 1.0482761
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482981, upper bound: 1.0482761
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0377284, upper bound: 1.0377284
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0377284, upper bound: 1.0377284
time: 0.33 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.10
Output dim: 0, lower bound: -1.0482981, upper bound: 1.0482761
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.10
Output dim: 0, lower bound: -1.0482981, upper bound: 1.0482761
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.10
Output dim: 0, lower bound: -1.0377284, upper bound: 1.0377284
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.10
Output dim: 0, lower bound: -1.0377284, upper bound: 1.0377284
Binary search (step 12): status=Status.VERIFIED, low=0.1999756, high=0.2000000, mid=0.1999756, abs_max=1.1883488893508911
rel_dist={0: [-1.0564885954033876, 1.0564885954033874]}

## Binary search (step 13) starts
Candidate diff: 0.1999878


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523240, upper bound: 1.0523240
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523240, upper bound: 1.0523240
time: 0.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.70 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.70
Output dim: 0, lower bound: -1.0523240, upper bound: 1.0523240
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.70
Output dim: 0, lower bound: -1.0523240, upper bound: 1.0523240

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510958, upper bound: 1.0513464
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510958, upper bound: 1.0510958
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0522922, upper bound: 1.0517134
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517134, upper bound: 1.0522922
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.10 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -1.0510958, upper bound: 1.0513464
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -1.0510958, upper bound: 1.0510958
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -1.0522922, upper bound: 1.0517134
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -1.0517134, upper bound: 1.0522922

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0486990, upper bound: 1.0486525
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0486990, upper bound: 1.0486525
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510958, upper bound: 1.0510663
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510663, upper bound: 1.0510958
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509764, upper bound: 1.0508944
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509764, upper bound: 1.0508589
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510914, upper bound: 1.0511147
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510914, upper bound: 1.0511147
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.18 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.18
Output dim: 0, lower bound: -1.0486990, upper bound: 1.0486525
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.18
Output dim: 0, lower bound: -1.0486990, upper bound: 1.0486525
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -1.0510958, upper bound: 1.0510663
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -1.0510663, upper bound: 1.0510958
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -1.0509764, upper bound: 1.0508944
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -1.0509764, upper bound: 1.0508589
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -1.0510914, upper bound: 1.0511147
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -1.0510914, upper bound: 1.0511147

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509484, upper bound: 1.0508433
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509484, upper bound: 1.0508433
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0485668, upper bound: 1.0486525
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0485668, upper bound: 1.0486525
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509764, upper bound: 1.0508944
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509396, upper bound: 1.0508640
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512383, upper bound: 1.0508589
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512438, upper bound: 1.0508470
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374935, upper bound: 1.0374935
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374935, upper bound: 1.0374935
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510914, upper bound: 1.0510754
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510810, upper bound: 1.0511147
time: 0.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.49 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -1.0509484, upper bound: 1.0508433
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -1.0509484, upper bound: 1.0508433
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.49
Output dim: 0, lower bound: -1.0485668, upper bound: 1.0486525
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.49
Output dim: 0, lower bound: -1.0485668, upper bound: 1.0486525
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -1.0509764, upper bound: 1.0508944
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -1.0509396, upper bound: 1.0508640
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -1.0512383, upper bound: 1.0508589
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -1.0512438, upper bound: 1.0508470
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.49
Output dim: 0, lower bound: -1.0374935, upper bound: 1.0374935
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.49
Output dim: 0, lower bound: -1.0374935, upper bound: 1.0374935
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -1.0510914, upper bound: 1.0510754
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -1.0510810, upper bound: 1.0511147

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507376, upper bound: 1.0504204
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507162, upper bound: 1.0505073
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0377284, upper bound: 1.0377284
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0377284, upper bound: 1.0377284
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501124, upper bound: 1.0501143
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501082, upper bound: 1.0501210
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501439, upper bound: 1.0500848
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501082, upper bound: 1.0500575
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505073, upper bound: 1.0504181
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507376, upper bound: 1.0504803
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507195, upper bound: 1.0501874
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507195, upper bound: 1.0505489
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0470636, upper bound: 1.0467075
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0470636, upper bound: 1.0467075
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504204, upper bound: 1.0507376
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507166, upper bound: 1.0506368
time: 0.41 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.31 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -1.0507376, upper bound: 1.0504204
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -1.0507162, upper bound: 1.0505073
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -1.0377284, upper bound: 1.0377284
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -1.0377284, upper bound: 1.0377284
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -1.0501124, upper bound: 1.0501143
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -1.0501082, upper bound: 1.0501210
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -1.0501439, upper bound: 1.0500848
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -1.0501082, upper bound: 1.0500575
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -1.0505073, upper bound: 1.0504181
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -1.0507376, upper bound: 1.0504803
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -1.0507195, upper bound: 1.0501874
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -1.0507195, upper bound: 1.0505489
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -1.0470636, upper bound: 1.0467075
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -1.0470636, upper bound: 1.0467075
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -1.0504204, upper bound: 1.0507376
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -1.0507166, upper bound: 1.0506368

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0468337, upper bound: 1.0464310
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0463886, upper bound: 1.0464310
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0468145, upper bound: 1.0464780
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0468145, upper bound: 1.0464780
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501093, upper bound: 1.0500515
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500819, upper bound: 1.0498621
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0473417, upper bound: 1.0476126
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480754, upper bound: 1.0476126
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0466001, upper bound: 1.0463886
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0464780, upper bound: 1.0463886
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504081, upper bound: 1.0507310
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504105, upper bound: 1.0504897
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0464310, upper bound: 1.0465634
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0463986, upper bound: 1.0465634
time: 0.34 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.28 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0468337, upper bound: 1.0464310
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0463886, upper bound: 1.0464310
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0468145, upper bound: 1.0464780
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0468145, upper bound: 1.0464780
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0501093, upper bound: 1.0500515
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0500819, upper bound: 1.0498621
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0473417, upper bound: 1.0476126
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0480754, upper bound: 1.0476126
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0466001, upper bound: 1.0463886
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0464780, upper bound: 1.0463886
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0504081, upper bound: 1.0507310
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0504105, upper bound: 1.0504897
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0464310, upper bound: 1.0465634
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 0, lower bound: -1.0463986, upper bound: 1.0465634

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0469980
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0463806, upper bound: 1.0468271
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0463806, upper bound: 1.0468271
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0464253, upper bound: 1.0465142
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0464253, upper bound: 1.0465142
time: 0.34 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.57 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0469980
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.0463806, upper bound: 1.0468271
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.0463806, upper bound: 1.0468271
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.0464253, upper bound: 1.0465142
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.0464253, upper bound: 1.0465142
Binary search (step 13): status=Status.VERIFIED, low=0.1999878, high=0.2000000, mid=0.1999878, abs_max=1.1883488893508911
rel_dist={0: [-1.0564886394127655, 1.0564886394127653]}

## Binary search (step 14) starts
Candidate diff: 0.1999939


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0564887, upper bound: 1.0564887
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0564887, upper bound: 1.0564887
time: 0.35 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.75 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -1.0564887, upper bound: 1.0564887
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -1.0564887, upper bound: 1.0564887

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0560075
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0544116
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0540170, upper bound: 1.0539330
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.10 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0560075
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0544116
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -1.0540170, upper bound: 1.0539330
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0540459, upper bound: 1.0540558
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0540459, upper bound: 1.0557692
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480179, upper bound: 1.0475060
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480179, upper bound: 1.0475060
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0537272, upper bound: 1.0536171
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536263, upper bound: 1.0536249
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536249, upper bound: 1.0536129
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536214, upper bound: 1.0537272
time: 0.35 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.11 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -1.0540459, upper bound: 1.0540558
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -1.0540459, upper bound: 1.0557692
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.11
Output dim: 0, lower bound: -1.0480179, upper bound: 1.0475060
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.11
Output dim: 0, lower bound: -1.0480179, upper bound: 1.0475060
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -1.0537272, upper bound: 1.0536171
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -1.0536263, upper bound: 1.0536249
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -1.0536249, upper bound: 1.0536129
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -1.0536214, upper bound: 1.0537272

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0540312, upper bound: 1.0540558
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0540693, upper bound: 1.0540331
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517232, upper bound: 1.0517951
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517251, upper bound: 1.0518128
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518128, upper bound: 1.0517251
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518128, upper bound: 1.0517251
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534652, upper bound: 1.0534525
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534773, upper bound: 1.0534646
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518068, upper bound: 1.0517232
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517951, upper bound: 1.0517232
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534582, upper bound: 1.0535439
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534582, upper bound: 1.0535664
time: 0.37 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -1.0540312, upper bound: 1.0540558
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -1.0540693, upper bound: 1.0540331
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -1.0517232, upper bound: 1.0517951
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -1.0517251, upper bound: 1.0518128
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -1.0518128, upper bound: 1.0517251
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -1.0518128, upper bound: 1.0517251
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -1.0534652, upper bound: 1.0534525
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -1.0534773, upper bound: 1.0534646
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -1.0518068, upper bound: 1.0517232
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -1.0517951, upper bound: 1.0517232
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -1.0534582, upper bound: 1.0535439
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -1.0534582, upper bound: 1.0535664

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0468781, upper bound: 1.0469722
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0468781, upper bound: 1.0469722
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0457430, upper bound: 1.0465260
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0457430, upper bound: 1.0465260
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0470377, upper bound: 1.0480754
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0470377, upper bound: 1.0480754
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374749, upper bound: 1.0374749
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374749, upper bound: 1.0374749
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501142, upper bound: 1.0500848
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501441, upper bound: 1.0500848
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480608, upper bound: 1.0480666
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480608, upper bound: 1.0480666
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505235
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501142, upper bound: 1.0500581
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501124, upper bound: 1.0500575
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501082, upper bound: 1.0500342
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501082, upper bound: 1.0498913
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
time: 0.34 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0468781, upper bound: 1.0469722
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0468781, upper bound: 1.0469722
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0457430, upper bound: 1.0465260
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0457430, upper bound: 1.0465260
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0470377, upper bound: 1.0480754
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0470377, upper bound: 1.0480754
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0374749, upper bound: 1.0374749
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0374749, upper bound: 1.0374749
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0501142, upper bound: 1.0500848
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0501441, upper bound: 1.0500848
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0480608, upper bound: 1.0480666
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0480608, upper bound: 1.0480666
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505235
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0501142, upper bound: 1.0500581
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0501124, upper bound: 1.0500575
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0501082, upper bound: 1.0500342
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0501082, upper bound: 1.0498913
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479866, upper bound: 1.0479854
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479866, upper bound: 1.0479854
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479862, upper bound: 1.0480804
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0473417, upper bound: 1.0478312
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0473417, upper bound: 1.0478312
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500808, upper bound: 1.0500047
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500812, upper bound: 1.0499796
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480754, upper bound: 1.0470377
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480754, upper bound: 1.0470377
time: 0.32 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.19 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.0479866, upper bound: 1.0479854
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.0479866, upper bound: 1.0479854
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.0479862, upper bound: 1.0480804
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.0473417, upper bound: 1.0478312
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.0473417, upper bound: 1.0478312
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.0500808, upper bound: 1.0500047
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.0500812, upper bound: 1.0499796
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.0480754, upper bound: 1.0470377
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.0480754, upper bound: 1.0470377

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.34 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.21 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.21
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
Binary search (step 14): status=Status.VERIFIED, low=0.1999939, high=0.2000000, mid=0.1999939, abs_max=1.1883488893508911
rel_dist={0: [-1.0564886613905864, 1.0564886613905862]}

## Binary search (step 15) starts
Candidate diff: 0.1999969


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0562181, upper bound: 1.0562181
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0562181, upper bound: 1.0563081
time: 0.31 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.66 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -1.0562181, upper bound: 1.0562181
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -1.0562181, upper bound: 1.0563081

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0545193, upper bound: 1.0551766
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0545193, upper bound: 1.0559249
time: 0.38 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0554830, upper bound: 1.0563081
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0556700, upper bound: 1.0563081
time: 0.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.21 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.21
Output dim: 0, lower bound: -1.0545193, upper bound: 1.0551766
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.21
Output dim: 0, lower bound: -1.0545193, upper bound: 1.0559249
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.21
Output dim: 0, lower bound: -1.0554830, upper bound: 1.0563081
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.21
Output dim: 0, lower bound: -1.0556700, upper bound: 1.0563081

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516770, upper bound: 1.0516805
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0522714, upper bound: 1.0516797
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544790, upper bound: 1.0553516
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544790, upper bound: 1.0550907
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0538300, upper bound: 1.0537705
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0537639, upper bound: 1.0538548
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478546, upper bound: 1.0466243
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478546, upper bound: 1.0466243
time: 0.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.10 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 0, lower bound: -1.0516770, upper bound: 1.0516805
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 0, lower bound: -1.0522714, upper bound: 1.0516797
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 0, lower bound: -1.0544790, upper bound: 1.0553516
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 0, lower bound: -1.0544790, upper bound: 1.0550907
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 0, lower bound: -1.0538300, upper bound: 1.0537705
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 0, lower bound: -1.0537639, upper bound: 1.0538548
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.10
Output dim: 0, lower bound: -1.0478546, upper bound: 1.0466243
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.10
Output dim: 0, lower bound: -1.0478546, upper bound: 1.0466243

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506077, upper bound: 1.0505669
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506077, upper bound: 1.0505669
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509498, upper bound: 1.0508690
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509498, upper bound: 1.0508299
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534520, upper bound: 1.0534526
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534520, upper bound: 1.0535483
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0538596, upper bound: 1.0547075
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0538581, upper bound: 1.0538802
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506102, upper bound: 1.0505569
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506102, upper bound: 1.0505569
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518484, upper bound: 1.0519048
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518430, upper bound: 1.0519048
time: 0.38 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -1.0506077, upper bound: 1.0505669
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -1.0506077, upper bound: 1.0505669
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -1.0509498, upper bound: 1.0508690
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -1.0509498, upper bound: 1.0508299
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -1.0534520, upper bound: 1.0534526
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -1.0534520, upper bound: 1.0535483
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -1.0538596, upper bound: 1.0547075
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -1.0538581, upper bound: 1.0538802
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -1.0506102, upper bound: 1.0505569
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -1.0506102, upper bound: 1.0505569
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -1.0518484, upper bound: 1.0519048
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -1.0518430, upper bound: 1.0519048

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0483873, upper bound: 1.0483616
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0483873, upper bound: 1.0483616
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505002, upper bound: 1.0505345
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505002, upper bound: 1.0505669
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506003, upper bound: 1.0507104
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506022, upper bound: 1.0507104
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501093, upper bound: 1.0500728
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500861, upper bound: 1.0498621
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504931
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504931
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0455292, upper bound: 1.0463296
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0455292, upper bound: 1.0463296
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479246, upper bound: 1.0484012
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479246, upper bound: 1.0484012
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0476395, upper bound: 1.0481922
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482772, upper bound: 1.0481917
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504380, upper bound: 1.0505123
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505837, upper bound: 1.0498599
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516426, upper bound: 1.0517202
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516319, upper bound: 1.0517173
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516426, upper bound: 1.0517202
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516319, upper bound: 1.0517173
time: 0.37 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.44 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0483873, upper bound: 1.0483616
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0483873, upper bound: 1.0483616
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0505002, upper bound: 1.0505345
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0505002, upper bound: 1.0505669
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0506003, upper bound: 1.0507104
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0506022, upper bound: 1.0507104
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0501093, upper bound: 1.0500728
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0500861, upper bound: 1.0498621
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504931
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504931
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0455292, upper bound: 1.0463296
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0455292, upper bound: 1.0463296
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0479246, upper bound: 1.0484012
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0479246, upper bound: 1.0484012
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0476395, upper bound: 1.0481922
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0482772, upper bound: 1.0481917
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0504380, upper bound: 1.0505123
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0505837, upper bound: 1.0498599
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0516426, upper bound: 1.0517202
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0516319, upper bound: 1.0517173
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0516426, upper bound: 1.0517202
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0516319, upper bound: 1.0517173

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0483873, upper bound: 1.0483616
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0483217, upper bound: 1.0483616
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0467049, upper bound: 1.0470591
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0467062, upper bound: 1.0470602
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504096, upper bound: 1.0507104
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504871, upper bound: 1.0507082
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0465285, upper bound: 1.0468067
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0465285, upper bound: 1.0468067
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480612, upper bound: 1.0469980
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480612, upper bound: 1.0469980
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503448, upper bound: 1.0503930
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503162, upper bound: 1.0504310
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505065, upper bound: 1.0494957
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503871, upper bound: 1.0498240
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500700, upper bound: 1.0501010
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500753, upper bound: 1.0498621
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500739, upper bound: 1.0500916
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500752, upper bound: 1.0498621
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374606, upper bound: 1.0374606
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374606, upper bound: 1.0374606
time: 0.35 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.43 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0483873, upper bound: 1.0483616
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0483217, upper bound: 1.0483616
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0467049, upper bound: 1.0470591
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0467062, upper bound: 1.0470602
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0504096, upper bound: 1.0507104
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0504871, upper bound: 1.0507082
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0465285, upper bound: 1.0468067
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0465285, upper bound: 1.0468067
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0480612, upper bound: 1.0469980
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0480612, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0374864, upper bound: 1.0374864
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0503448, upper bound: 1.0503930
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0503162, upper bound: 1.0504310
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0505065, upper bound: 1.0494957
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0503871, upper bound: 1.0498240
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0500700, upper bound: 1.0501010
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0500753, upper bound: 1.0498621
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0500739, upper bound: 1.0500916
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0500752, upper bound: 1.0498621
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0374606, upper bound: 1.0374606
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0374606, upper bound: 1.0374606

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0464376, upper bound: 1.0467852
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0464376, upper bound: 1.0467852
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0476019
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480724, upper bound: 1.0469980
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480724, upper bound: 1.0469980
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480724, upper bound: 1.0469980
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480724, upper bound: 1.0469980
time: 0.37 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.29 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.29
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.29
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.29
Output dim: 0, lower bound: -1.0464376, upper bound: 1.0467852
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.29
Output dim: 0, lower bound: -1.0464376, upper bound: 1.0467852
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.29
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0476019
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.29
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.29
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.29
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.29
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.29
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.29
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.29
Output dim: 0, lower bound: -1.0374784, upper bound: 1.0374784
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.29
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.29
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.29
Output dim: 0, lower bound: -1.0480724, upper bound: 1.0469980
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.29
Output dim: 0, lower bound: -1.0480724, upper bound: 1.0469980
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.29
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.29
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.29
Output dim: 0, lower bound: -1.0480724, upper bound: 1.0469980
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.29
Output dim: 0, lower bound: -1.0480724, upper bound: 1.0469980
Binary search (step 15): status=Status.VERIFIED, low=0.1999969, high=0.2000000, mid=0.1999969, abs_max=1.1883488893508911
rel_dist={0: [-1.0564886724063647, 1.0564886724063651]}

## Binary search (step 16) starts
Candidate diff: 0.1999985


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0538449, upper bound: 1.0538449
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0538449, upper bound: 1.0538449
time: 0.37 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -1.0538449, upper bound: 1.0538449
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -1.0538449, upper bound: 1.0538449

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0533842, upper bound: 1.0535611
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535611, upper bound: 1.0533885
time: 0.39 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0481199, upper bound: 1.0481199
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0481199, upper bound: 1.0481199
time: 0.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.22 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -1.0533842, upper bound: 1.0535611
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -1.0535611, upper bound: 1.0533885
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.22
Output dim: 0, lower bound: -1.0481199, upper bound: 1.0481199
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.22
Output dim: 0, lower bound: -1.0481199, upper bound: 1.0481199

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0466243, upper bound: 1.0478546
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0466243, upper bound: 1.0478546
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0533842, upper bound: 1.0533797
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0533328, upper bound: 1.0533885
time: 0.36 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.56 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.56
Output dim: 0, lower bound: -1.0466243, upper bound: 1.0478546
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.56
Output dim: 0, lower bound: -1.0466243, upper bound: 1.0478546
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -1.0533842, upper bound: 1.0533797
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -1.0533328, upper bound: 1.0533885

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478444, upper bound: 1.0466234
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478444, upper bound: 1.0466234
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0477437, upper bound: 1.0466234
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0477437, upper bound: 1.0466234
time: 0.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.23 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.23
Output dim: 0, lower bound: -1.0478444, upper bound: 1.0466234
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.23
Output dim: 0, lower bound: -1.0478444, upper bound: 1.0466234
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.23
Output dim: 0, lower bound: -1.0477437, upper bound: 1.0466234
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.23
Output dim: 0, lower bound: -1.0477437, upper bound: 1.0466234
Binary search (step 16): status=Status.VERIFIED, low=0.1999985, high=0.2000000, mid=0.1999985, abs_max=1.1883488893508911
rel_dist={0: [-1.0564886779411216, 1.0564886779411218]}

## Binary search (step 17) starts
Candidate diff: 0.1999992


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0564887, upper bound: 1.0564887
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0564887, upper bound: 1.0564887
time: 0.35 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -1.0564887, upper bound: 1.0564887
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -1.0564887, upper bound: 1.0564887

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0560075
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0544116
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0560075
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0544116
time: 0.32 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.20 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0560075
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0544116
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0560075
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0544116

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0542186, upper bound: 1.0553007
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0542186, upper bound: 1.0558319
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480179, upper bound: 1.0475060
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480179, upper bound: 1.0475060
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544033, upper bound: 1.0559897
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0544029
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519933, upper bound: 1.0519586
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519933, upper bound: 1.0519667
time: 0.33 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.19 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.0542186, upper bound: 1.0553007
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.0542186, upper bound: 1.0558319
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.0480179, upper bound: 1.0475060
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.0480179, upper bound: 1.0475060
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.0544033, upper bound: 1.0559897
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.0544116, upper bound: 1.0544029
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.0519933, upper bound: 1.0519586
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.0519933, upper bound: 1.0519667

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0538596, upper bound: 1.0538942
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0538407, upper bound: 1.0549452
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0541488, upper bound: 1.0558109
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0542186, upper bound: 1.0541445
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508433, upper bound: 1.0509484
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508433, upper bound: 1.0509484
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0540476, upper bound: 1.0540331
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0540346, upper bound: 1.0540317
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518128, upper bound: 1.0517251
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518153, upper bound: 1.0517762
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518423, upper bound: 1.0518497
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518978, upper bound: 1.0518761
time: 0.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0538596, upper bound: 1.0538942
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0538407, upper bound: 1.0549452
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0541488, upper bound: 1.0558109
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0542186, upper bound: 1.0541445
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0508433, upper bound: 1.0509484
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0508433, upper bound: 1.0509484
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0540476, upper bound: 1.0540331
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0540346, upper bound: 1.0540317
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0518128, upper bound: 1.0517251
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0518153, upper bound: 1.0517762
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0518423, upper bound: 1.0518497
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0518978, upper bound: 1.0518761

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517001, upper bound: 1.0517171
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516806, upper bound: 1.0517171
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482965, upper bound: 1.0483661
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482965, upper bound: 1.0483661
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0379707, upper bound: 1.0379707
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0379707, upper bound: 1.0379707
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0465209, upper bound: 1.0465209
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0465209, upper bound: 1.0465209
time: 0.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505073, upper bound: 1.0507153
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504803, upper bound: 1.0507376
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0470093, upper bound: 1.0471668
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0470093, upper bound: 1.0471668
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505073, upper bound: 1.0507162
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505073, upper bound: 1.0507162
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0457430, upper bound: 1.0458952
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0457430, upper bound: 1.0458952
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374749, upper bound: 1.0374749
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374749, upper bound: 1.0374749
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480103, upper bound: 1.0481114
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480103, upper bound: 1.0481114
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0379707, upper bound: 1.0379707
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0379707, upper bound: 1.0379707
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0379707, upper bound: 1.0379707
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0379707, upper bound: 1.0379707
time: 0.39 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.40 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0517001, upper bound: 1.0517171
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0516806, upper bound: 1.0517171
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0482965, upper bound: 1.0483661
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0482965, upper bound: 1.0483661
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0379707, upper bound: 1.0379707
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0379707, upper bound: 1.0379707
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0465209, upper bound: 1.0465209
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0465209, upper bound: 1.0465209
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0505073, upper bound: 1.0507153
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0504803, upper bound: 1.0507376
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0470093, upper bound: 1.0471668
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0470093, upper bound: 1.0471668
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0505073, upper bound: 1.0507162
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0505073, upper bound: 1.0507162
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0457430, upper bound: 1.0458952
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0457430, upper bound: 1.0458952
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0374749, upper bound: 1.0374749
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0374749, upper bound: 1.0374749
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0480103, upper bound: 1.0481114
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0480103, upper bound: 1.0481114
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0379707, upper bound: 1.0379707
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0379707, upper bound: 1.0379707
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0379707, upper bound: 1.0379707
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.40
Output dim: 0, lower bound: -1.0379707, upper bound: 1.0379707

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374606, upper bound: 1.0374606
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374606, upper bound: 1.0374606
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374606, upper bound: 1.0374606
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374606, upper bound: 1.0374606
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0463886, upper bound: 1.0468337
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0463886, upper bound: 1.0468337
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504883, upper bound: 1.0507099
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504983, upper bound: 1.0501750
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0464780, upper bound: 1.0468145
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0464780, upper bound: 1.0468145
time: 0.38 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.43 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0374606, upper bound: 1.0374606
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0374606, upper bound: 1.0374606
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0374606, upper bound: 1.0374606
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0374606, upper bound: 1.0374606
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0374702, upper bound: 1.0374702
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0463886, upper bound: 1.0468337
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0463886, upper bound: 1.0468337
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0504883, upper bound: 1.0507099
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0504983, upper bound: 1.0501750
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0464780, upper bound: 1.0468145
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.43
Output dim: 0, lower bound: -1.0464780, upper bound: 1.0468145

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0463677, upper bound: 1.0468067
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0463677, upper bound: 1.0468067
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
time: 0.35 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.74 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.74
Output dim: 0, lower bound: -1.0463677, upper bound: 1.0468067
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.74
Output dim: 0, lower bound: -1.0463677, upper bound: 1.0468067
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.74
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.74
Output dim: 0, lower bound: -1.0374424, upper bound: 1.0374424
Binary search (step 17): status=Status.VERIFIED, low=0.1999992, high=0.2000000, mid=0.1999992, abs_max=1.1883488893508911
rel_dist={0: [-1.0564886834221432, 1.0564886834221427]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.1999992251396634
execution time: 1086.50 seconds
