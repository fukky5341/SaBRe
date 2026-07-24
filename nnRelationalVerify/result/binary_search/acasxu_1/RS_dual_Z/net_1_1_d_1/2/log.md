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
execution time: IAR + LP analysis = 1.57 + 1.13 = 2.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1.0564887, upper bound: 1.0564887


# Binary Search by BASE starts (time budget: 1197.30 seconds, max iter: 100)

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
Binary search time: 48.98 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1148.32 seconds

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

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
- Time for RS candidates: 0.81 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.36 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0506509
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507328
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0506509
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507328
time: 0.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.44 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0506509
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507328
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0506509
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507328

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506223, upper bound: 1.0506150
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506150
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506162
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0505224
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506149, upper bound: 1.0505830
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0506150
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506150
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0506162
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506197, upper bound: 1.0505224
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0505582
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0505830
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506223
time: 0.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -1.0506223, upper bound: 1.0506150
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506150
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506162
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0505224
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -1.0506149, upper bound: 1.0505830
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0506150
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506150
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0506162
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -1.0506197, upper bound: 1.0505224
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0505582
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0505830
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506223

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0505185
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504719
time: 0.35 seconds

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0504931
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505189
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0496554, upper bound: 1.0506057
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504699, upper bound: 1.0506057
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504656, upper bound: 1.0504578
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504656, upper bound: 1.0504607
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494887, upper bound: 1.0505689
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505563, upper bound: 1.0505689
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494887, upper bound: 1.0506077
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505345, upper bound: 1.0506077
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504607, upper bound: 1.0505185
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504719
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504357, upper bound: 1.0504932
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0505189
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505191
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0500394
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504656, upper bound: 1.0504578
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505189, upper bound: 1.0504607
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0494047
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505185, upper bound: 1.0505232
time: 0.36 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.97 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0505185
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504719
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0504931
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505189
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0496554, upper bound: 1.0506057
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0504699, upper bound: 1.0506057
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0504656, upper bound: 1.0504578
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0504656, upper bound: 1.0504607
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0494887, upper bound: 1.0505689
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0505563, upper bound: 1.0505689
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0494887, upper bound: 1.0506077
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0505345, upper bound: 1.0506077
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0504607, upper bound: 1.0505185
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504719
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0504357, upper bound: 1.0504932
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0505189
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505191
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0500394
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0504656, upper bound: 1.0504578
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0505189, upper bound: 1.0504607
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0494047
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 0, lower bound: -1.0505185, upper bound: 1.0505232

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474069, upper bound: 1.0480451
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480451
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480399
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0493246, upper bound: 1.0505113
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494957, upper bound: 1.0505065
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500173, upper bound: 1.0505113
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503930, upper bound: 1.0505065
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474069, upper bound: 1.0476019
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
time: 0.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505113, upper bound: 1.0500173
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505113, upper bound: 1.0493246
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503162, upper bound: 1.0504310
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0493246, upper bound: 1.0498240
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504729, upper bound: 1.0504345
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504729, upper bound: 1.0503581
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503448, upper bound: 1.0504759
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0493246, upper bound: 1.0493246
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504519, upper bound: 1.0504759
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504513, upper bound: 1.0493246
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0493246, upper bound: 1.0505143
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0493246, upper bound: 1.0505075
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503895, upper bound: 1.0505143
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504499, upper bound: 1.0505075
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480451
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0479862
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0474510
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480399
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0474069
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0476019
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
time: 0.38 seconds

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0479678
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480332, upper bound: 1.0479717
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0469980
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479862, upper bound: 1.0480804
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0480951
time: 0.38 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.97 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0474069, upper bound: 1.0480451
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480451
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480399
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0493246, upper bound: 1.0505113
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0494957, upper bound: 1.0505065
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0500173, upper bound: 1.0505113
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0503930, upper bound: 1.0505065
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0474069, upper bound: 1.0476019
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0505113, upper bound: 1.0500173
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0505113, upper bound: 1.0493246
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0503162, upper bound: 1.0504310
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0493246, upper bound: 1.0498240
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0504729, upper bound: 1.0504345
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0504729, upper bound: 1.0503581
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0503448, upper bound: 1.0504759
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0493246, upper bound: 1.0493246
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0504519, upper bound: 1.0504759
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0504513, upper bound: 1.0493246
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0493246, upper bound: 1.0505143
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0493246, upper bound: 1.0505075
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0503895, upper bound: 1.0505143
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0504499, upper bound: 1.0505075
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480451
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0479862
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0474510
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480399
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0474069
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0476019
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0479678
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0480332, upper bound: 1.0479717
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0479862, upper bound: 1.0480804
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0480951

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480724
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480724
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480673
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480752
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0469980
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0475947
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0475853
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474069, upper bound: 1.0469980
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480724, upper bound: 1.0469980
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479678
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0473028
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479914, upper bound: 1.0479710
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475752, upper bound: 1.0479558
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479137
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475752, upper bound: 1.0475772
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478402, upper bound: 1.0479386
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0469980
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480332, upper bound: 1.0479717
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480332, upper bound: 1.0469980
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480612
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480612
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480593
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480419
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479862, upper bound: 1.0480804
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.38 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.93 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480724
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480724
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480673
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480752
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0475947
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0475853
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0474069, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0480724, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479678
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0473028
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0479914, upper bound: 1.0479710
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0475752, upper bound: 1.0479558
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479137
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0475752, upper bound: 1.0475772
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0478402, upper bound: 1.0479386
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0480332, upper bound: 1.0479717
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0480332, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480612
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480612
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480593
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480419
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0479862, upper bound: 1.0480804
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
Binary search (step 0): status=Status.VERIFIED, low=0.1000000, high=0.2000000, mid=0.1000000, abs_max=1.1883488893508911
rel_dist={0: [-1.0558835892060525, 1.0558835892060525]}

## Binary search (step 1) starts
Candidate diff: 0.1500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.78 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

## BFS RS instance: RS_RSZ1

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
time: 0.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507328, upper bound: 1.0507156
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
time: 0.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.23 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -1.0507328, upper bound: 1.0507156
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
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
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0506509
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0506509
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328
time: 0.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.31 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0506509
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0506509
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0506150
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506150
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506162
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0505224
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0505582
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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0505830
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506223
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506223, upper bound: 1.0506150
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506150
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506162
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0505224
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0505830
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223
time: 0.39 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0506150
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506150
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506162
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0505224
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0505582
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0505830
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506223
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0506223, upper bound: 1.0506150
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506150
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506162
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0505224
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0505830
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505185
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504719
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504931
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505189
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505234
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504357, upper bound: 1.0505191
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504932, upper bound: 1.0504357
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504656, upper bound: 1.0504578
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505189, upper bound: 1.0504607
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505164, upper bound: 1.0494047
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505232
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504607, upper bound: 1.0505185
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504719
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504931
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504607, upper bound: 1.0505189
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505191
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0500394
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504656, upper bound: 1.0504578
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505189, upper bound: 1.0504607
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505164, upper bound: 1.0494047
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505232
time: 0.38 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505185
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504719
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504931
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505189
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505234
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504357, upper bound: 1.0505191
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504932, upper bound: 1.0504357
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504656, upper bound: 1.0504578
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0505189, upper bound: 1.0504607
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0505164, upper bound: 1.0494047
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505232
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504607, upper bound: 1.0505185
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504719
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504931
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504607, upper bound: 1.0505189
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505191
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0500394
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504656, upper bound: 1.0504578
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0505189, upper bound: 1.0504607
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0505164, upper bound: 1.0494047
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505232

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480451
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0474589
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480399
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0469980
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474069, upper bound: 1.0476019
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0493246, upper bound: 1.0504310
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503871, upper bound: 1.0498240
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504729, upper bound: 1.0504345
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504729, upper bound: 1.0503581
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0469980
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0479717
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0493246, upper bound: 1.0505143
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503895, upper bound: 1.0505143
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0493246, upper bound: 1.0505075
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504499, upper bound: 1.0505075
time: 0.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480451
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0474510
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480399
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0474069
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0476019
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0475947
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0479678
time: 0.37 seconds

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480332, upper bound: 1.0479717
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.35 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480451
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0474589
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480399
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0474069, upper bound: 1.0476019
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0493246, upper bound: 1.0504310
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0503871, upper bound: 1.0498240
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0504729, upper bound: 1.0504345
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0504729, upper bound: 1.0503581
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0479717
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0493246, upper bound: 1.0505143
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0503895, upper bound: 1.0505143
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0493246, upper bound: 1.0505075
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0504499, upper bound: 1.0505075
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480451
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0474510
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480399
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0474069
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0476019
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0475947
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0479678
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0480332, upper bound: 1.0479717
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475752, upper bound: 1.0479678
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0473028
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479914, upper bound: 1.0479710
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479513, upper bound: 1.0479558
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479137
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0475772
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480612
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480612
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479862, upper bound: 1.0480804
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480593
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480419
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.35 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.77 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.77
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.77
Output dim: 0, lower bound: -1.0475752, upper bound: 1.0479678
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.77
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.77
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0473028
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.77
Output dim: 0, lower bound: -1.0479914, upper bound: 1.0479710
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.77
Output dim: 0, lower bound: -1.0479513, upper bound: 1.0479558
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.77
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479137
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.77
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0475772
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.77
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480612
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.77
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480612
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.77
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.77
Output dim: 0, lower bound: -1.0479862, upper bound: 1.0480804
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.77
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480593
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.77
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480419
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.77
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.77
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
Binary search (step 1): status=Status.VERIFIED, low=0.1500000, high=0.2000000, mid=0.1500000, abs_max=1.1883488893508911
rel_dist={0: [-1.0562380918366323, 1.056238091836632]}

## Binary search (step 2) starts
Candidate diff: 0.1750000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

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
- Time for RS candidates: 0.82 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

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
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
time: 0.34 seconds

## BFS RS instance: RS_RSZ2

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.26 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507079
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0506509
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0506509
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328
time: 0.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.33 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507079
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0506509
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0506509
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506150
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506150
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506162
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506197, upper bound: 1.0505224
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506149, upper bound: 1.0505830
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506223, upper bound: 1.0506150
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506150
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506162
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505224
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506149, upper bound: 1.0505830
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223
time: 0.40 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506150
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506150
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506162
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -1.0506197, upper bound: 1.0505224
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -1.0506149, upper bound: 1.0505830
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -1.0506223, upper bound: 1.0506150
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506150
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506162
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505224
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -1.0506149, upper bound: 1.0505830
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505232, upper bound: 1.0505185
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504719
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0504932
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505189
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504357, upper bound: 1.0505191
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0504578
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505189, upper bound: 1.0504607
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505164, upper bound: 1.0494047
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505232
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505232, upper bound: 1.0505185
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504719
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504931
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504607, upper bound: 1.0505189
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504357, upper bound: 1.0505191
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0504578
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505189, upper bound: 1.0504607
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505164, upper bound: 1.0494047
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505185, upper bound: 1.0505232
time: 0.32 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.86 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0505232, upper bound: 1.0505185
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504719
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0504932
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505189
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0504357, upper bound: 1.0505191
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0504578
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0505189, upper bound: 1.0504607
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0505164, upper bound: 1.0494047
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505232
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0505232, upper bound: 1.0505185
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504719
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504931
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0504607, upper bound: 1.0505189
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0504357, upper bound: 1.0505191
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0504578
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0505189, upper bound: 1.0504607
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0505164, upper bound: 1.0494047
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 0, lower bound: -1.0505185, upper bound: 1.0505232

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480399
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0469980
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474069, upper bound: 1.0476019
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
time: 0.38 seconds

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0479678
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0469980
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0479717
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505075, upper bound: 1.0504499
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505075, upper bound: 1.0493246
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0474510
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480399
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0474069
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0476019
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
time: 0.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0479678
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480332, upper bound: 1.0479717
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479862, upper bound: 1.0480804
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.38 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.03 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480399
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0474069, upper bound: 1.0476019
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0479678
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0479717
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0505075, upper bound: 1.0504499
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0505075, upper bound: 1.0493246
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0474510
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480399
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0474069
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0476019
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0479678
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480332, upper bound: 1.0479717
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0479862, upper bound: 1.0480804
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480419, upper bound: 1.0469980
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480593, upper bound: 1.0469980
time: 0.34 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.23 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.23
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.23
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.23
Output dim: 0, lower bound: -1.0480419, upper bound: 1.0469980
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.23
Output dim: 0, lower bound: -1.0480593, upper bound: 1.0469980
Binary search (step 2): status=Status.VERIFIED, low=0.1750000, high=0.2000000, mid=0.1750000, abs_max=1.1883488893508911
rel_dist={0: [-1.0563961018081798, 1.0563961018081796]}

## Binary search (step 3) starts
Candidate diff: 0.1875000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.36 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.89 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.89
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.89
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507328, upper bound: 1.0507156
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
time: 0.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
time: 0.37 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.55 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 0, lower bound: -1.0507328, upper bound: 1.0507156
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0506509
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507328
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507079
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0506509
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328
time: 0.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.52 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0506509
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507328
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507079
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0506509
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506223, upper bound: 1.0506150
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506149
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506162
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506197, upper bound: 1.0505224
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0505830
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223
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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0506150
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506149
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506162
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506197, upper bound: 1.0505224
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506149, upper bound: 1.0505830
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506223
time: 0.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0506223, upper bound: 1.0506150
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506149
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506162
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0506197, upper bound: 1.0505224
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0505830
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0506150
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506149
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506162
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0506197, upper bound: 1.0505224
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0506149, upper bound: 1.0505830
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506223

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505232, upper bound: 1.0505185
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504719
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504931
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504607, upper bound: 1.0505189
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
time: 0.37 seconds

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504357, upper bound: 1.0505191
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505164, upper bound: 1.0500394
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504656, upper bound: 1.0504578
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505189, upper bound: 1.0504607
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505164, upper bound: 1.0494047
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505185, upper bound: 1.0505232
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505232, upper bound: 1.0505185
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504719
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504931
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504607, upper bound: 1.0505189
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504357, upper bound: 1.0505191
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0504578
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505189, upper bound: 1.0504607
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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505164, upper bound: 1.0494047
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505235
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505185, upper bound: 1.0505232
time: 0.33 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0505232, upper bound: 1.0505185
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504719
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504931
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0504607, upper bound: 1.0505189
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0504357, upper bound: 1.0505191
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0505164, upper bound: 1.0500394
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0504656, upper bound: 1.0504578
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0505189, upper bound: 1.0504607
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0505164, upper bound: 1.0494047
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0505185, upper bound: 1.0505232
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0505232, upper bound: 1.0505185
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504719
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504931
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0504607, upper bound: 1.0505189
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0504357, upper bound: 1.0505191
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0504578
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0505189, upper bound: 1.0504607
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0505164, upper bound: 1.0494047
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505235
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.79
Output dim: 0, lower bound: -1.0505185, upper bound: 1.0505232

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480451
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480399
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0469980
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474069, upper bound: 1.0476019
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0479678
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0469980
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480332, upper bound: 1.0479717
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479862, upper bound: 1.0480804
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505075, upper bound: 1.0504499
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505075, upper bound: 1.0493246
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505143, upper bound: 1.0503895
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505143, upper bound: 1.0493246
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0474510
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480399
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0474069
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0476019
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0479678
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480332, upper bound: 1.0479717
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479862, upper bound: 1.0480804
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.37 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480451
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480399
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0474069, upper bound: 1.0476019
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0479678
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480332, upper bound: 1.0479717
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479862, upper bound: 1.0480804
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0505075, upper bound: 1.0504499
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0505075, upper bound: 1.0493246
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0505143, upper bound: 1.0503895
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0505143, upper bound: 1.0493246
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0474510
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480399
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0474069
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0476019
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0479678
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480332, upper bound: 1.0479717
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479862, upper bound: 1.0480804
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480419, upper bound: 1.0469980
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480593, upper bound: 1.0469980
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0473028, upper bound: 1.0479862
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480612, upper bound: 1.0469980
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480612, upper bound: 1.0469980
time: 0.34 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.75 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 0, lower bound: -1.0480419, upper bound: 1.0469980
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 0, lower bound: -1.0480593, upper bound: 1.0469980
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 0, lower bound: -1.0473028, upper bound: 1.0479862
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 0, lower bound: -1.0480612, upper bound: 1.0469980
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.75
Output dim: 0, lower bound: -1.0480612, upper bound: 1.0469980
Binary search (step 3): status=Status.VERIFIED, low=0.1875000, high=0.2000000, mid=0.1875000, abs_max=1.1883488893508911
rel_dist={0: [-1.0564436068034628, 1.0564436068034624]}

## Binary search (step 4) starts
Candidate diff: 0.1937500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.35 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.84 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.84
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.84
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
time: 0.34 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
time: 0.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.30 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0506509
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507079
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0506509
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328
time: 0.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0506509
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507079
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0506509
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506150
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506149
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0506162
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505224
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506149, upper bound: 1.0505830
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506223, upper bound: 1.0506150
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506150
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506162
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
time: 0.36 seconds

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505224
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506149, upper bound: 1.0505830
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223
time: 0.39 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.33 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506150
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506149
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0506162
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505224
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.0506149, upper bound: 1.0505830
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.0506223, upper bound: 1.0506150
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506150
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506162
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505224
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.0506149, upper bound: 1.0505830
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505232, upper bound: 1.0505185
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504719
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504931
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504607, upper bound: 1.0505189
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504357, upper bound: 1.0505191
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504656, upper bound: 1.0504578
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505189, upper bound: 1.0504607
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505164, upper bound: 1.0494047
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505185, upper bound: 1.0505232
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505232, upper bound: 1.0505185
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504719
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504931
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505189
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504357, upper bound: 1.0505191
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504656, upper bound: 1.0504578
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505189, upper bound: 1.0504607
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505164, upper bound: 1.0494047
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505185, upper bound: 1.0505232
time: 0.33 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0505232, upper bound: 1.0505185
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504719
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504931
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504607, upper bound: 1.0505189
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504357, upper bound: 1.0505191
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504656, upper bound: 1.0504578
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0505189, upper bound: 1.0504607
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0505164, upper bound: 1.0494047
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0505185, upper bound: 1.0505232
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0505232, upper bound: 1.0505185
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504719
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504931
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505189
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504357, upper bound: 1.0505191
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504656, upper bound: 1.0504578
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0505189, upper bound: 1.0504607
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0505164, upper bound: 1.0494047
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 0, lower bound: -1.0505185, upper bound: 1.0505232

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480451
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0480451
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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480399
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0469980
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474069, upper bound: 1.0476019
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0479678
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0479710
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0469980
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0479717
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0469980
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503581, upper bound: 1.0504499
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505075, upper bound: 1.0493246
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0474510
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480399
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0474069
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0476019
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0475947
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0479678
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479717
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480951
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.36 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480451
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0480451
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480399
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0474069, upper bound: 1.0476019
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0479678
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0479710
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0479717
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0503581, upper bound: 1.0504499
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0505075, upper bound: 1.0493246
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0474510
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480399
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0474069
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0476019
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0475947
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0479678
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479717
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480951
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.92
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480419, upper bound: 1.0480451
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480419, upper bound: 1.0469980
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480419, upper bound: 1.0469980
time: 0.38 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.84 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -1.0480419, upper bound: 1.0480451
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -1.0480419, upper bound: 1.0469980
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -1.0480419, upper bound: 1.0469980
Binary search (step 4): status=Status.VERIFIED, low=0.1937500, high=0.2000000, mid=0.1937500, abs_max=1.1883488893508911
rel_dist={0: [-1.0564661451396706, 1.0564661451396704]}

## Binary search (step 5) starts
Candidate diff: 0.1968750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

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
time: 0.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.81 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.29 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
time: 0.35 seconds

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0506509
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0506509
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328
time: 0.35 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.34 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0506509
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0506509
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506150
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506149
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0506162
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0505224
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

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506149, upper bound: 1.0505830
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506150
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506150
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0506162
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0505224
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506149, upper bound: 1.0505830
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223
time: 0.39 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506150
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506149
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0506162
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0505224
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0506149, upper bound: 1.0505830
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506150
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506150
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0506162
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0505224
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0506149, upper bound: 1.0505830
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.37
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505232, upper bound: 1.0505185
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504719
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504931
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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504607, upper bound: 1.0505189
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505191
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504656, upper bound: 1.0504578
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504607
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0494047
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505185, upper bound: 1.0505232
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505232, upper bound: 1.0505185
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504719
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504931
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504607, upper bound: 1.0505189
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
time: 0.38 seconds

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505234
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504357, upper bound: 1.0505191
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0504578
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505189, upper bound: 1.0504607
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0494047
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505185, upper bound: 1.0505232
time: 0.33 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.81 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0505232, upper bound: 1.0505185
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504719
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504931
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0504607, upper bound: 1.0505189
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505191
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0504656, upper bound: 1.0504578
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504607
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0494047
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0505185, upper bound: 1.0505232
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0505232, upper bound: 1.0505185
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504719
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504931
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0504607, upper bound: 1.0505189
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505234
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0504357, upper bound: 1.0505191
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0504578
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0505189, upper bound: 1.0504607
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0494047
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -1.0505185, upper bound: 1.0505232

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474069, upper bound: 1.0480451
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480451
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0474589
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.35 seconds

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480399
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0469980
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474069, upper bound: 1.0476019
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0479678
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0469980
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480332, upper bound: 1.0479717
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0469980
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
time: 0.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503581, upper bound: 1.0504499
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505075, upper bound: 1.0493246
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0474510
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480399
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0474069
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0476019
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0479678
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480332, upper bound: 1.0479717
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.36 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0474069, upper bound: 1.0480451
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480451
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0474589
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480399
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0474069, upper bound: 1.0476019
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0479678
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480332, upper bound: 1.0479717
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0503581, upper bound: 1.0504499
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0505075, upper bound: 1.0493246
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0474510
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480399
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0474069
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0476019
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0479678
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480332, upper bound: 1.0479717
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480419, upper bound: 1.0469980
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0469980
time: 0.34 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.85 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.85
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.85
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.85
Output dim: 0, lower bound: -1.0480419, upper bound: 1.0469980
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.85
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0469980
Binary search (step 5): status=Status.VERIFIED, low=0.1968750, high=0.2000000, mid=0.1968750, abs_max=1.1883488893508911
rel_dist={0: [-1.0564774142809072, 1.056477414280907]}

## Binary search (step 6) starts
Candidate diff: 0.1984375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

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
- Time for RS candidates: 0.80 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507328, upper bound: 1.0507156
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.29 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -1.0507328, upper bound: 1.0507156
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
time: 0.35 seconds

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0506509
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328
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
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0506509
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328
time: 0.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.47 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0506509
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0506509
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328
Binary search (step 6): status=Status.UNKNOWN, low=0.1968750, high=0.1984375, mid=0.1984375, abs_max=1.1883488893508911
rel_dist={0: [-1.0564830488246575, 1.0564830488246573]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.19687498826533556
execution time: 1148.99 seconds
