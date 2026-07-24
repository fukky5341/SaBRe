## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_2.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 2.7638016924


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331)
1: (-0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803)
2: (-1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662)
3: (-1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606)
4: (-1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884)

## BASE Result
execution time: IAR + LP analysis = 1.44 + 1.04 = 2.48 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2.7804846, upper bound: 2.7804846


# Binary Search by BASE starts (time budget: 1197.52 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1000000, mid=0.1000000, abs_max=3.285133123397827
rel_dist={0: [-2.7804845941221, 2.7804845941221004]}

## Binary search (step 1) starts
Candidate diff: 0.0500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0500000, mid=0.0500000, abs_max=3.285133123397827
rel_dist={0: [-2.7803829052250393, 2.7803829052250393]}

## Binary search (step 2) starts
Candidate diff: 0.0250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0250000, mid=0.0250000, abs_max=3.285133123397827
rel_dist={0: [-2.780286344644136, 2.7802863446441357]}

## Binary search (step 3) starts
Candidate diff: 0.0125000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0125000, mid=0.0125000, abs_max=3.285133123397827
rel_dist={0: [-2.7801994131262022, 2.7801994131262022]}

## Binary search (step 4) starts
Candidate diff: 0.0062500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0062500, mid=0.0062500, abs_max=3.285133123397827
rel_dist={0: [-2.78013659937945, 2.78013659937945]}

## Binary search (step 5) starts
Candidate diff: 0.0031250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0031250, mid=0.0031250, abs_max=3.285133123397827
rel_dist={0: [-2.7790340042848785, 2.7790340042848776]}

## Binary search (step 6) starts
Candidate diff: 0.0015625


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0015625, mid=0.0015625, abs_max=3.285133123397827
rel_dist={0: [-2.7780414768623047, 2.778041476862305]}

## Binary search (step 7) starts
Candidate diff: 0.0007812


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0007812, mid=0.0007812, abs_max=3.285133123397827
rel_dist={0: [-2.777472476768272, 2.7774724767682724]}

## Binary search (step 8) starts
Candidate diff: 0.0003906


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0003906, mid=0.0003906, abs_max=3.285133123397827
rel_dist={0: [-2.7771819898984202, 2.7771819898984216]}

## Binary search (step 9) starts
Candidate diff: 0.0001953


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0001953, mid=0.0001953, abs_max=3.285133123397827
rel_dist={0: [-2.7770315308237956, 2.7770315308237947]}

## Binary search (step 10) starts
Candidate diff: 0.0000977


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0000977, mid=0.0000977, abs_max=3.285133123397827
rel_dist={0: [-2.77695630129431, 2.7769563012943106]}

## Binary search (step 11) starts
Candidate diff: 0.0000488


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000488, mid=0.0000488, abs_max=3.285133123397827
rel_dist={0: [-2.776918686545108, 2.776918686545107]}

## Binary search (step 12) starts
Candidate diff: 0.0000244


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000244, mid=0.0000244, abs_max=3.285133123397827
rel_dist={0: [-2.7768998792011317, 2.7768998792011335]}

## Binary search (step 13) starts
Candidate diff: 0.0000122


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000122, mid=0.0000122, abs_max=3.285133123397827
rel_dist={0: [-2.776890475588643, 2.776890475588642]}

## Binary search (step 14) starts
Candidate diff: 0.0000061


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000061, mid=0.0000061, abs_max=3.285133123397827
rel_dist={0: [-2.776885819152771, 2.7768857738947865]}

## Binary search (step 15) starts
Candidate diff: 0.0000031


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000031, mid=0.0000031, abs_max=3.285133123397827
rel_dist={0: [-2.7768834397858853, 2.7768834397858857]}

## Binary search (step 16) starts
Candidate diff: 0.0000015


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000015, mid=0.0000015, abs_max=3.285133123397827
rel_dist={0: [-2.776882250874421, 2.776882259450736]}

## Binary search (step 17) starts
Candidate diff: 0.0000008


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000008, mid=0.0000008, abs_max=3.285133123397827
rel_dist={0: [-2.7768821313393013, 2.776881713387178]}

## Binary Search Result
Binary search time: 45.04 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1152.48 seconds

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804502
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804673
time: 0.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804502
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804673

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7793461, upper bound: 2.7796655
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7793461, upper bound: 2.7796655
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801151, upper bound: 2.7801380
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800837, upper bound: 2.7801380
time: 0.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.03 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -2.7793461, upper bound: 2.7796655
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -2.7793461, upper bound: 2.7796655
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -2.7801151, upper bound: 2.7801380
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -2.7800837, upper bound: 2.7801380

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756370, upper bound: 2.7793940
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791934, upper bound: 2.7794346
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7783030, upper bound: 2.7792720
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789162, upper bound: 2.7754943
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801151, upper bound: 2.7800073
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7801380
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800155, upper bound: 2.7801380
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800837, upper bound: 2.7768811
time: 0.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.07 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 0, lower bound: -2.7756370, upper bound: 2.7793940
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 0, lower bound: -2.7791934, upper bound: 2.7794346
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 0, lower bound: -2.7783030, upper bound: 2.7792720
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 0, lower bound: -2.7789162, upper bound: 2.7754943
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 0, lower bound: -2.7801151, upper bound: 2.7800073
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7801380
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 0, lower bound: -2.7800155, upper bound: 2.7801380
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 0, lower bound: -2.7800837, upper bound: 2.7768811

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748125, upper bound: 2.7789702
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748104, upper bound: 2.7783349
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791934, upper bound: 2.7794346
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756370, upper bound: 2.7792196
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7783030, upper bound: 2.7789670
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782184, upper bound: 2.7792720
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751637, upper bound: 2.7753317
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7787676, upper bound: 2.7753216
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799277, upper bound: 2.7793418
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801151, upper bound: 2.7800073
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7790476
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7776028
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800155, upper bound: 2.7768811
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7798526, upper bound: 2.7801380
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7632881, upper bound: 2.7586014
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7632881, upper bound: 2.7586014
time: 0.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.09 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7748125, upper bound: 2.7789702
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7748104, upper bound: 2.7783349
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7791934, upper bound: 2.7794346
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7756370, upper bound: 2.7792196
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7783030, upper bound: 2.7789670
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7782184, upper bound: 2.7792720
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7751637, upper bound: 2.7753317
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7787676, upper bound: 2.7753216
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7799277, upper bound: 2.7793418
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7801151, upper bound: 2.7800073
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7790476
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7776028
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7800155, upper bound: 2.7768811
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7798526, upper bound: 2.7801380
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7632881, upper bound: 2.7586014
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7632881, upper bound: 2.7586014

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748125, upper bound: 2.7757856
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748104, upper bound: 2.7749799
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748104, upper bound: 2.7755494
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748104, upper bound: 2.7748690
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776028, upper bound: 2.7755129
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791934, upper bound: 2.7794346
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7586014, upper bound: 2.7598735
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7586014, upper bound: 2.7610807
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7783030, upper bound: 2.7748442
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782512, upper bound: 2.7789670
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749227, upper bound: 2.7759051
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755987, upper bound: 2.7750250
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748104, upper bound: 2.7752469
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750198, upper bound: 2.7748481
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786160, upper bound: 2.7746755
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7787676, upper bound: 2.7753216
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7784014
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7776492
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800947, upper bound: 2.7799066
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800336, upper bound: 2.7799838
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7786880
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7790476
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7755647
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7776028
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769860, upper bound: 2.7768811
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782610, upper bound: 2.7768811
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7576881, upper bound: 2.7600098
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577029, upper bound: 2.7581324
time: 0.40 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7748125, upper bound: 2.7757856
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7748104, upper bound: 2.7749799
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7748104, upper bound: 2.7755494
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7748104, upper bound: 2.7748690
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7776028, upper bound: 2.7755129
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7791934, upper bound: 2.7794346
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7586014, upper bound: 2.7598735
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7586014, upper bound: 2.7610807
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7783030, upper bound: 2.7748442
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7782512, upper bound: 2.7789670
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7749227, upper bound: 2.7759051
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7755987, upper bound: 2.7750250
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7748104, upper bound: 2.7752469
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7750198, upper bound: 2.7748481
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7786160, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7787676, upper bound: 2.7753216
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7784014
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7776492
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7800947, upper bound: 2.7799066
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7800336, upper bound: 2.7799838
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7786880
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7790476
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7755647
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7776028
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7769860, upper bound: 2.7768811
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7782610, upper bound: 2.7768811
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7576881, upper bound: 2.7600098
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7577029, upper bound: 2.7581324

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746887
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747324
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7576076, upper bound: 2.7576933
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7576076, upper bound: 2.7576933
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757665, upper bound: 2.7765276
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7764157, upper bound: 2.7757080
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782794, upper bound: 2.7746755
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7585270, upper bound: 2.7562794
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7585270, upper bound: 2.7564021
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7756822
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748198, upper bound: 2.7756822
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748104, upper bound: 2.7752469
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748104, upper bound: 2.7752467
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748104, upper bound: 2.7748451
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750198, upper bound: 2.7748481
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747621, upper bound: 2.7746755
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754917, upper bound: 2.7746755
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749221, upper bound: 2.7751748
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755682, upper bound: 2.7746771
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757096, upper bound: 2.7758785
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7755129
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769265, upper bound: 2.7768811
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7781379, upper bound: 2.7775950
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753216, upper bound: 2.7787676
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7783621, upper bound: 2.7787676
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577225, upper bound: 2.7549012
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577225, upper bound: 2.7549012
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7762559
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7755590
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7574694, upper bound: 2.7617151
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7574694, upper bound: 2.7617151
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747076
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7576933, upper bound: 2.7576076
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7576933, upper bound: 2.7576076
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757142, upper bound: 2.7755129
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757142, upper bound: 2.7755129
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782610, upper bound: 2.7768811
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782445, upper bound: 2.7768811
time: 0.37 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746887
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747324
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7576076, upper bound: 2.7576933
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7576076, upper bound: 2.7576933
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7757665, upper bound: 2.7765276
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7764157, upper bound: 2.7757080
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7782794, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7585270, upper bound: 2.7562794
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7585270, upper bound: 2.7564021
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7756822
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7748198, upper bound: 2.7756822
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7748104, upper bound: 2.7752469
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7748104, upper bound: 2.7752467
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7748104, upper bound: 2.7748451
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7750198, upper bound: 2.7748481
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7747621, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7754917, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7749221, upper bound: 2.7751748
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7755682, upper bound: 2.7746771
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7757096, upper bound: 2.7758785
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7755129
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7769265, upper bound: 2.7768811
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7781379, upper bound: 2.7775950
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7753216, upper bound: 2.7787676
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7783621, upper bound: 2.7787676
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7577225, upper bound: 2.7549012
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7577225, upper bound: 2.7549012
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7762559
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7755590
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7574694, upper bound: 2.7617151
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7574694, upper bound: 2.7617151
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747076
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7576933, upper bound: 2.7576076
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7576933, upper bound: 2.7576076
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7757142, upper bound: 2.7755129
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7757142, upper bound: 2.7755129
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7782610, upper bound: 2.7768811
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -2.7782445, upper bound: 2.7768811

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748198, upper bound: 2.7756812
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7756822
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7751748
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749218, upper bound: 2.7746779
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746776
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749221, upper bound: 2.7751748
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747043
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755682, upper bound: 2.7746771
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746771
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748544, upper bound: 2.7746755
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756822, upper bound: 2.7746755
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756822, upper bound: 2.7746755
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746771, upper bound: 2.7755682
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751748, upper bound: 2.7749221
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7566529, upper bound: 2.7532883
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7575015, upper bound: 2.7533279
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535629, upper bound: 2.7533225
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535629, upper bound: 2.7533225
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532991
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532625
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751734, upper bound: 2.7746755
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754335, upper bound: 2.7746755
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756822, upper bound: 2.7746755
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756822, upper bound: 2.7746755
time: 0.37 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.27 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7748198, upper bound: 2.7756812
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7756822
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7751748
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7749218, upper bound: 2.7746779
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746776
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7749221, upper bound: 2.7751748
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747043
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7755682, upper bound: 2.7746771
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746771
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7748544, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7756822, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7756822, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7746771, upper bound: 2.7755682
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7751748, upper bound: 2.7749221
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7566529, upper bound: 2.7532883
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7575015, upper bound: 2.7533279
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7535629, upper bound: 2.7533225
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7535629, upper bound: 2.7533225
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532991
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7751734, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7754335, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7756822, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.27
Output dim: 0, lower bound: -2.7756822, upper bound: 2.7746755

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7476016, upper bound: 2.7475631
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7476016, upper bound: 2.7475631
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.33 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.27 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7476016, upper bound: 2.7475631
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7476016, upper bound: 2.7475631
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
Binary search (step 0): status=Status.VERIFIED, low=0.1000000, high=0.2000000, mid=0.1000000, abs_max=3.285133123397827
rel_dist={0: [-2.7804845941221, 2.7804845941221004]}

## Binary search (step 1) starts
Candidate diff: 0.1500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804502
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804673
time: 0.36 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804502
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804673

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804673, upper bound: 2.7800772
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800772, upper bound: 2.7804502
time: 0.37 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769176, upper bound: 2.7769176
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769176, upper bound: 2.7769176
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.07 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -2.7804673, upper bound: 2.7800772
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -2.7800772, upper bound: 2.7804502
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -2.7769176, upper bound: 2.7769176
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -2.7769176, upper bound: 2.7769176

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800772, upper bound: 2.7771527
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804673, upper bound: 2.7800772
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772216, upper bound: 2.7788129
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791401, upper bound: 2.7776674
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770332, upper bound: 2.7765091
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770176, upper bound: 2.7736505
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7633037, upper bound: 2.7624790
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7633037, upper bound: 2.7590368
time: 0.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.23 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.23
Output dim: 0, lower bound: -2.7800772, upper bound: 2.7771527
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.23
Output dim: 0, lower bound: -2.7804673, upper bound: 2.7800772
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.23
Output dim: 0, lower bound: -2.7772216, upper bound: 2.7788129
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.23
Output dim: 0, lower bound: -2.7791401, upper bound: 2.7776674
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.23
Output dim: 0, lower bound: -2.7770332, upper bound: 2.7765091
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.23
Output dim: 0, lower bound: -2.7770176, upper bound: 2.7736505
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.23
Output dim: 0, lower bound: -2.7633037, upper bound: 2.7624790
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.23
Output dim: 0, lower bound: -2.7633037, upper bound: 2.7590368

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7777243, upper bound: 2.7756882
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7792044, upper bound: 2.7756882
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7623851, upper bound: 2.7599663
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7623851, upper bound: 2.7603169
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772216, upper bound: 2.7788129
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7780466, upper bound: 2.7787987
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791401, upper bound: 2.7772483
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7787572, upper bound: 2.7776674
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7617325, upper bound: 2.7624790
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7615891, upper bound: 2.7615414
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7763408, upper bound: 2.7733070
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7767665, upper bound: 2.7732651
time: 0.38 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -2.7777243, upper bound: 2.7756882
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -2.7792044, upper bound: 2.7756882
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.15
Output dim: 0, lower bound: -2.7623851, upper bound: 2.7599663
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.15
Output dim: 0, lower bound: -2.7623851, upper bound: 2.7603169
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -2.7772216, upper bound: 2.7788129
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -2.7780466, upper bound: 2.7787987
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -2.7791401, upper bound: 2.7772483
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -2.7787572, upper bound: 2.7776674
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.15
Output dim: 0, lower bound: -2.7617325, upper bound: 2.7624790
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.15
Output dim: 0, lower bound: -2.7615891, upper bound: 2.7615414
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -2.7763408, upper bound: 2.7733070
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.15
Output dim: 0, lower bound: -2.7767665, upper bound: 2.7732651

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753979, upper bound: 2.7748442
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770801, upper bound: 2.7748442
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7783030, upper bound: 2.7748442
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7787363, upper bound: 2.7748442
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7780299, upper bound: 2.7787796
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772313, upper bound: 2.7788129
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756882, upper bound: 2.7767430
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756882, upper bound: 2.7767430
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577242, upper bound: 2.7545876
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577242, upper bound: 2.7545807
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7765600, upper bound: 2.7729847
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7767665, upper bound: 2.7732651
time: 0.39 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.13 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -2.7753979, upper bound: 2.7748442
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -2.7770801, upper bound: 2.7748442
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -2.7783030, upper bound: 2.7748442
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -2.7787363, upper bound: 2.7748442
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -2.7780299, upper bound: 2.7787796
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -2.7772313, upper bound: 2.7788129
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -2.7756882, upper bound: 2.7767430
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -2.7756882, upper bound: 2.7767430
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -2.7577242, upper bound: 2.7545876
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -2.7577242, upper bound: 2.7545807
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -2.7765600, upper bound: 2.7729847
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -2.7767665, upper bound: 2.7732651

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752518, upper bound: 2.7746755
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7563340, upper bound: 2.7563104
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7562794, upper bound: 2.7563104
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782794, upper bound: 2.7746755
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748902, upper bound: 2.7748442
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757012, upper bound: 2.7748442
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748442, upper bound: 2.7759051
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749252, upper bound: 2.7759051
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748442, upper bound: 2.7758889
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748442, upper bound: 2.7753548
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7736078, upper bound: 2.7729847
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751164, upper bound: 2.7726669
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7570709, upper bound: 2.7545645
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7595022, upper bound: 2.7545645
time: 0.37 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -2.7752518, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.15
Output dim: 0, lower bound: -2.7563340, upper bound: 2.7563104
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.15
Output dim: 0, lower bound: -2.7562794, upper bound: 2.7563104
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -2.7782794, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -2.7748902, upper bound: 2.7748442
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -2.7757012, upper bound: 2.7748442
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -2.7748442, upper bound: 2.7759051
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -2.7749252, upper bound: 2.7759051
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.15
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.15
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.15
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.15
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -2.7748442, upper bound: 2.7758889
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -2.7748442, upper bound: 2.7753548
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -2.7736078, upper bound: 2.7729847
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.15
Output dim: 0, lower bound: -2.7751164, upper bound: 2.7726669
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.15
Output dim: 0, lower bound: -2.7570709, upper bound: 2.7545645
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.15
Output dim: 0, lower bound: -2.7595022, upper bound: 2.7545645

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750833, upper bound: 2.7746755
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747621, upper bound: 2.7746755
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754917, upper bound: 2.7746755
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7755711
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747986, upper bound: 2.7756812
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7756822
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7756822
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7751734
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747043
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.38 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7750833, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7747621, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7754917, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7755711
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7747986, upper bound: 2.7756812
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7756822
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7756822
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7751734
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747043
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.19
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.18
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
Binary search (step 1): status=Status.VERIFIED, low=0.1500000, high=0.2000000, mid=0.1500000, abs_max=3.285133123397827
rel_dist={0: [-2.7804845941221, 2.7804845941221004]}

## Binary search (step 2) starts
Candidate diff: 0.1750000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770346, upper bound: 2.7770346
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770346, upper bound: 2.7770346
time: 0.35 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -2.7770346, upper bound: 2.7770346
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -2.7770346, upper bound: 2.7770346

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7740142, upper bound: 2.7754294
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7740142, upper bound: 2.7740142
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768750, upper bound: 2.7770346
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770346, upper bound: 2.7768776
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.02 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.02
Output dim: 0, lower bound: -2.7740142, upper bound: 2.7754294
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.02
Output dim: 0, lower bound: -2.7740142, upper bound: 2.7740142
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.02
Output dim: 0, lower bound: -2.7768750, upper bound: 2.7770346
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.02
Output dim: 0, lower bound: -2.7770346, upper bound: 2.7768776

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521671, upper bound: 2.7521671
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521671, upper bound: 2.7521671
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7740142, upper bound: 2.7740045
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754294, upper bound: 2.7740142
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7765787, upper bound: 2.7767684
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760603, upper bound: 2.7766792
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769176, upper bound: 2.7768685
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769176, upper bound: 2.7768761
time: 0.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.08 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.08
Output dim: 0, lower bound: -2.7521671, upper bound: 2.7521671
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.08
Output dim: 0, lower bound: -2.7521671, upper bound: 2.7521671
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -2.7740142, upper bound: 2.7740045
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -2.7754294, upper bound: 2.7740142
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -2.7765787, upper bound: 2.7767684
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -2.7760603, upper bound: 2.7766792
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -2.7769176, upper bound: 2.7768685
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -2.7769176, upper bound: 2.7768761

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749963, upper bound: 2.7739187
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752230, upper bound: 2.7736922
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750868, upper bound: 2.7739194
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752942, upper bound: 2.7735876
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7732725, upper bound: 2.7767684
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7761444, upper bound: 2.7767046
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7745646, upper bound: 2.7763549
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758949, upper bound: 2.7726705
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7735171, upper bound: 2.7767261
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7765091, upper bound: 2.7766112
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7594931, upper bound: 2.7638474
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7594931, upper bound: 2.7638474
time: 0.34 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.08 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -2.7749963, upper bound: 2.7739187
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -2.7752230, upper bound: 2.7736922
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -2.7750868, upper bound: 2.7739194
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -2.7752942, upper bound: 2.7735876
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -2.7732725, upper bound: 2.7767684
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -2.7761444, upper bound: 2.7767046
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -2.7745646, upper bound: 2.7763549
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -2.7758949, upper bound: 2.7726705
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -2.7735171, upper bound: 2.7767261
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -2.7765091, upper bound: 2.7766112
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -2.7594931, upper bound: 2.7638474
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -2.7594931, upper bound: 2.7638474

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521671, upper bound: 2.7521619
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521671, upper bound: 2.7521619
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751238, upper bound: 2.7736908
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752216, upper bound: 2.7735171
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7726705, upper bound: 2.7732666
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747323, upper bound: 2.7735706
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750830, upper bound: 2.7735862
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752927, upper bound: 2.7735623
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7732651, upper bound: 2.7767665
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7732704, upper bound: 2.7757998
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7761272, upper bound: 2.7766988
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7761114, upper bound: 2.7730454
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7569060, upper bound: 2.7565553
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7565209, upper bound: 2.7588544
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7579128, upper bound: 2.7565173
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7565173, upper bound: 2.7565173
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7735171, upper bound: 2.7752216
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7735171, upper bound: 2.7739079
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7615263, upper bound: 2.7613112
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7624629, upper bound: 2.7611867
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582293, upper bound: 2.7568751
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7579160, upper bound: 2.7601078
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7594785, upper bound: 2.7638474
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7594931, upper bound: 2.7600239
time: 0.37 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.14 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7521671, upper bound: 2.7521619
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7521671, upper bound: 2.7521619
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7751238, upper bound: 2.7736908
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7752216, upper bound: 2.7735171
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7726705, upper bound: 2.7732666
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7747323, upper bound: 2.7735706
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7750830, upper bound: 2.7735862
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7752927, upper bound: 2.7735623
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7732651, upper bound: 2.7767665
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7732704, upper bound: 2.7757998
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7761272, upper bound: 2.7766988
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7761114, upper bound: 2.7730454
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7569060, upper bound: 2.7565553
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7565209, upper bound: 2.7588544
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7579128, upper bound: 2.7565173
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7565173, upper bound: 2.7565173
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7735171, upper bound: 2.7752216
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7735171, upper bound: 2.7739079
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7615263, upper bound: 2.7613112
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7624629, upper bound: 2.7611867
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7582293, upper bound: 2.7568751
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7579160, upper bound: 2.7601078
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7594785, upper bound: 2.7638474
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7594931, upper bound: 2.7600239

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7562498, upper bound: 2.7562498
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7562498, upper bound: 2.7562498
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749535, upper bound: 2.7726669
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751164, upper bound: 2.7726669
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7528613, upper bound: 2.7528613
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7528613, upper bound: 2.7528613
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7730454, upper bound: 2.7735450
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747153, upper bound: 2.7735633
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749042, upper bound: 2.7728539
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746347, upper bound: 2.7728236
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749686, upper bound: 2.7726669
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751875, upper bound: 2.7728063
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7545645, upper bound: 2.7595022
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7545645, upper bound: 2.7595022
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7728236, upper bound: 2.7746347
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7732704, upper bound: 2.7728520
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7601078, upper bound: 2.7562794
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7601078, upper bound: 2.7566422
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7580445, upper bound: 2.7562794
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7564302, upper bound: 2.7562794
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7726669, upper bound: 2.7751164
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7726669, upper bound: 2.7749535
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7562498, upper bound: 2.7562498
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7562498, upper bound: 2.7562498
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7574694, upper bound: 2.7617151
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7574694, upper bound: 2.7574694
time: 0.36 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.18 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7562498, upper bound: 2.7562498
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7562498, upper bound: 2.7562498
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7749535, upper bound: 2.7726669
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7751164, upper bound: 2.7726669
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7528613, upper bound: 2.7528613
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7528613, upper bound: 2.7528613
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7730454, upper bound: 2.7735450
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7747153, upper bound: 2.7735633
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7749042, upper bound: 2.7728539
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7746347, upper bound: 2.7728236
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7749686, upper bound: 2.7726669
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7751875, upper bound: 2.7728063
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7545645, upper bound: 2.7595022
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7545645, upper bound: 2.7595022
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7728236, upper bound: 2.7746347
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7732704, upper bound: 2.7728520
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7601078, upper bound: 2.7562794
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7601078, upper bound: 2.7566422
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7580445, upper bound: 2.7562794
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7564302, upper bound: 2.7562794
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7726669, upper bound: 2.7751164
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7726669, upper bound: 2.7749535
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7562498, upper bound: 2.7562498
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7562498, upper bound: 2.7562498
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7574694, upper bound: 2.7617151
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 0, lower bound: -2.7574694, upper bound: 2.7574694

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.32 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.10 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
Binary search (step 2): status=Status.VERIFIED, low=0.1750000, high=0.2000000, mid=0.1750000, abs_max=3.285133123397827
rel_dist={0: [-2.7804845941221, 2.7804845941221004]}

## Binary search (step 3) starts
Candidate diff: 0.1875000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804502
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804673
time: 0.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.71 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.71
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804502
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.71
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804673

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7803401
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7804290
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776703, upper bound: 2.7791509
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788967, upper bound: 2.7780636
time: 0.37 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.05 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.05
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7803401
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.05
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7804290
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.05
Output dim: 0, lower bound: -2.7776703, upper bound: 2.7791509
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.05
Output dim: 0, lower bound: -2.7788967, upper bound: 2.7780636

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7780588, upper bound: 2.7788018
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791509, upper bound: 2.7772483
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772321, upper bound: 2.7788967
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7787863, upper bound: 2.7776703
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776674, upper bound: 2.7791401
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772216, upper bound: 2.7791401
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788967, upper bound: 2.7772321
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788018, upper bound: 2.7780588
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.15 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.15
Output dim: 0, lower bound: -2.7780588, upper bound: 2.7788018
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.15
Output dim: 0, lower bound: -2.7791509, upper bound: 2.7772483
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.15
Output dim: 0, lower bound: -2.7772321, upper bound: 2.7788967
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.15
Output dim: 0, lower bound: -2.7787863, upper bound: 2.7776703
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.15
Output dim: 0, lower bound: -2.7776674, upper bound: 2.7791401
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.15
Output dim: 0, lower bound: -2.7772216, upper bound: 2.7791401
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.15
Output dim: 0, lower bound: -2.7788967, upper bound: 2.7772321
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.15
Output dim: 0, lower bound: -2.7788018, upper bound: 2.7780588

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7780390, upper bound: 2.7786448
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7780390, upper bound: 2.7787796
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7785020, upper bound: 2.7769614
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7785020, upper bound: 2.7769571
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7783191
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769982, upper bound: 2.7783318
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752101, upper bound: 2.7738643
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741893, upper bound: 2.7738643
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772439, upper bound: 2.7791401
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776674, upper bound: 2.7791222
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754231, upper bound: 2.7751424
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757353, upper bound: 2.7751424
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760085, upper bound: 2.7750410
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760085, upper bound: 2.7750277
time: 0.41 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 0, lower bound: -2.7780390, upper bound: 2.7786448
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 0, lower bound: -2.7780390, upper bound: 2.7787796
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 0, lower bound: -2.7785020, upper bound: 2.7769614
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 0, lower bound: -2.7785020, upper bound: 2.7769571
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7783191
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 0, lower bound: -2.7769982, upper bound: 2.7783318
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 0, lower bound: -2.7752101, upper bound: 2.7738643
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 0, lower bound: -2.7741893, upper bound: 2.7738643
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 0, lower bound: -2.7772439, upper bound: 2.7791401
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 0, lower bound: -2.7776674, upper bound: 2.7791222
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.23
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.23
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 0, lower bound: -2.7754231, upper bound: 2.7751424
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 0, lower bound: -2.7757353, upper bound: 2.7751424
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 0, lower bound: -2.7760085, upper bound: 2.7750410
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.23
Output dim: 0, lower bound: -2.7760085, upper bound: 2.7750277

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7775897, upper bound: 2.7781126
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7775931, upper bound: 2.7771159
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7780299, upper bound: 2.7787796
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7780390, upper bound: 2.7787755
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748104, upper bound: 2.7749799
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754830, upper bound: 2.7749799
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7784448, upper bound: 2.7769571
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7785020, upper bound: 2.7769361
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7740847, upper bound: 2.7738643
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7726669, upper bound: 2.7738372
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776674, upper bound: 2.7787572
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772483, upper bound: 2.7791222
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752469, upper bound: 2.7750259
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752469, upper bound: 2.7748104
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755494, upper bound: 2.7750259
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755494, upper bound: 2.7748104
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757856, upper bound: 2.7749314
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757856, upper bound: 2.7748583
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759948, upper bound: 2.7749887
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760085, upper bound: 2.7750277
time: 0.39 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7775897, upper bound: 2.7781126
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7775931, upper bound: 2.7771159
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7780299, upper bound: 2.7787796
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7780390, upper bound: 2.7787755
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7748104, upper bound: 2.7749799
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7754830, upper bound: 2.7749799
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7784448, upper bound: 2.7769571
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7785020, upper bound: 2.7769361
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7740847, upper bound: 2.7738643
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7726669, upper bound: 2.7738372
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7776674, upper bound: 2.7787572
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7772483, upper bound: 2.7791222
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7752469, upper bound: 2.7750259
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7752469, upper bound: 2.7748104
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7755494, upper bound: 2.7750259
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7755494, upper bound: 2.7748104
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7757856, upper bound: 2.7749314
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7757856, upper bound: 2.7748583
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7759948, upper bound: 2.7749887
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -2.7760085, upper bound: 2.7750277

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7775897, upper bound: 2.7781126
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7773257, upper bound: 2.7768811
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7775931, upper bound: 2.7771159
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7782712
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776045, upper bound: 2.7782712
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7782445
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7775950, upper bound: 2.7781379
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748104, upper bound: 2.7749006
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748104, upper bound: 2.7749799
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753794, upper bound: 2.7748663
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7748590
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7784448, upper bound: 2.7768811
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7784344, upper bound: 2.7769571
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748458, upper bound: 2.7757972
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748713, upper bound: 2.7757972
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755412, upper bound: 2.7748104
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755494, upper bound: 2.7748104
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751164, upper bound: 2.7726669
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749163, upper bound: 2.7726669
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751875, upper bound: 2.7728063
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751875, upper bound: 2.7728224
time: 0.38 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7775897, upper bound: 2.7781126
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7773257, upper bound: 2.7768811
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7775931, upper bound: 2.7771159
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7782712
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7776045, upper bound: 2.7782712
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7782445
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7775950, upper bound: 2.7781379
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7748104, upper bound: 2.7749006
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7748104, upper bound: 2.7749799
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7753794, upper bound: 2.7748663
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7748590
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7784448, upper bound: 2.7768811
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7784344, upper bound: 2.7769571
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7748458, upper bound: 2.7757972
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7748713, upper bound: 2.7757972
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7755412, upper bound: 2.7748104
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7755494, upper bound: 2.7748104
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7524068, upper bound: 2.7524068
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7751164, upper bound: 2.7726669
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7749163, upper bound: 2.7726669
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7751875, upper bound: 2.7728063
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -2.7751875, upper bound: 2.7728224

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747986, upper bound: 2.7756812
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748198, upper bound: 2.7756812
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7756822
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7756822
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7756822
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7756822
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7748663
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7748590
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753794, upper bound: 2.7748663
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750833, upper bound: 2.7746755
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754294, upper bound: 2.7746755
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751591, upper bound: 2.7748525
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754137, upper bound: 2.7748525
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747278, upper bound: 2.7755682
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747294, upper bound: 2.7746755
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
time: 0.36 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7747986, upper bound: 2.7756812
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7748198, upper bound: 2.7756812
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7756822
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7756822
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7756822
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7756822
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7748663
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7748590
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7753794, upper bound: 2.7748663
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7750833, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7754294, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7751591, upper bound: 2.7748525
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7754137, upper bound: 2.7748525
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7747278, upper bound: 2.7755682
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7747294, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 0, lower bound: -2.7523912, upper bound: 2.7523912

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.39 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.31
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
Binary search (step 3): status=Status.VERIFIED, low=0.1875000, high=0.2000000, mid=0.1875000, abs_max=3.285133123397827
rel_dist={0: [-2.7804845941221, 2.7804845941221004]}

## Binary search (step 4) starts
Candidate diff: 0.1937500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804846, upper bound: 2.7803453
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803453, upper bound: 2.7804846
time: 0.35 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.71 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.71
Output dim: 0, lower bound: -2.7804846, upper bound: 2.7803453
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.71
Output dim: 0, lower bound: -2.7803453, upper bound: 2.7804846

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801481, upper bound: 2.7801481
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801481, upper bound: 2.7803346
time: 0.34 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771918, upper bound: 2.7801671
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800387, upper bound: 2.7801671
time: 0.37 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.05 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.05
Output dim: 0, lower bound: -2.7801481, upper bound: 2.7801481
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.05
Output dim: 0, lower bound: -2.7801481, upper bound: 2.7803346
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.05
Output dim: 0, lower bound: -2.7771918, upper bound: 2.7801671
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.05
Output dim: 0, lower bound: -2.7800387, upper bound: 2.7801671

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755926, upper bound: 2.7790627
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7785380, upper bound: 2.7790627
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800561, upper bound: 2.7800152
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800555, upper bound: 2.7768956
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768956, upper bound: 2.7785311
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771918, upper bound: 2.7777063
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7598270, upper bound: 2.7618610
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7598270, upper bound: 2.7618610
time: 0.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.08 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -2.7755926, upper bound: 2.7790627
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -2.7785380, upper bound: 2.7790627
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -2.7800561, upper bound: 2.7800152
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -2.7800555, upper bound: 2.7768956
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -2.7768956, upper bound: 2.7785311
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -2.7771918, upper bound: 2.7777063
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.08
Output dim: 0, lower bound: -2.7598270, upper bound: 2.7618610
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.08
Output dim: 0, lower bound: -2.7598270, upper bound: 2.7618610

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749676, upper bound: 2.7759671
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754785, upper bound: 2.7751908
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7784898, upper bound: 2.7789087
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7784321, upper bound: 2.7747947
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769723, upper bound: 2.7784624
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782816, upper bound: 2.7776754
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582386, upper bound: 2.7540471
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582386, upper bound: 2.7540471
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768956, upper bound: 2.7785311
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768956, upper bound: 2.7784697
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768956, upper bound: 2.7769799
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768956, upper bound: 2.7776495
time: 0.37 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -2.7749676, upper bound: 2.7759671
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -2.7754785, upper bound: 2.7751908
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -2.7784898, upper bound: 2.7789087
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -2.7784321, upper bound: 2.7747947
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -2.7769723, upper bound: 2.7784624
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -2.7782816, upper bound: 2.7776754
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -2.7582386, upper bound: 2.7540471
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -2.7582386, upper bound: 2.7540471
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -2.7768956, upper bound: 2.7785311
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -2.7768956, upper bound: 2.7784697
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -2.7768956, upper bound: 2.7769799
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -2.7768956, upper bound: 2.7776495

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754649, upper bound: 2.7749654
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754785, upper bound: 2.7751908
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7779868, upper bound: 2.7788146
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7783621, upper bound: 2.7787676
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7783942, upper bound: 2.7747947
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7784321, upper bound: 2.7747947
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7770397
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769571, upper bound: 2.7784344
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521619
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521619
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756619, upper bound: 2.7766353
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756619, upper bound: 2.7766353
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768956, upper bound: 2.7769741
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768956, upper bound: 2.7769799
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7748975
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7747971
time: 0.36 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -2.7754649, upper bound: 2.7749654
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -2.7754785, upper bound: 2.7751908
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -2.7779868, upper bound: 2.7788146
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -2.7783621, upper bound: 2.7787676
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -2.7783942, upper bound: 2.7747947
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -2.7784321, upper bound: 2.7747947
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7770397
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -2.7769571, upper bound: 2.7784344
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521619
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521619
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -2.7756619, upper bound: 2.7766353
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -2.7756619, upper bound: 2.7766353
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -2.7768956, upper bound: 2.7769741
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -2.7768956, upper bound: 2.7769799
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7748975
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.08
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7747971

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752482, upper bound: 2.7748442
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753548, upper bound: 2.7748442
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7756246
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753794, upper bound: 2.7748663
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7566688, upper bound: 2.7533047
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7575184, upper bound: 2.7533446
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7581533, upper bound: 2.7537457
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7581533, upper bound: 2.7537457
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748907, upper bound: 2.7747947
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755600, upper bound: 2.7747947
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521619
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521619
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7749456
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7747947
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7769614
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7769635
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7748907
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7748975
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.34 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7752482, upper bound: 2.7748442
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7753548, upper bound: 2.7748442
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7756246
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7753794, upper bound: 2.7748663
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7566688, upper bound: 2.7533047
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7575184, upper bound: 2.7533446
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7581533, upper bound: 2.7537457
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7581533, upper bound: 2.7537457
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7748907, upper bound: 2.7747947
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7755600, upper bound: 2.7747947
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521619
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521619
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7749456
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7747947
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7769614
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7769635
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7748907
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7748975
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.06
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750833, upper bound: 2.7746755
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747043, upper bound: 2.7746755
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751734, upper bound: 2.7746755
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7756246
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747091, upper bound: 2.7746755
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747294, upper bound: 2.7746755
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747294
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747091
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.36 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.14 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7750833, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7747043, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7751734, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7756246
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7747091, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7747294, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747294
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747091
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.14
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.33 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.13 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
Binary search (step 4): status=Status.VERIFIED, low=0.1937500, high=0.2000000, mid=0.1937500, abs_max=3.285133123397827
rel_dist={0: [-2.7804845941221, 2.7804845941221004]}

## Binary search (step 5) starts
Candidate diff: 0.1968750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
time: 0.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.69 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.69
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.69
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7794964, upper bound: 2.7794558
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7794558, upper bound: 2.7794964
time: 0.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770190, upper bound: 2.7770346
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770190, upper bound: 2.7770346
time: 0.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.01 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.01
Output dim: 0, lower bound: -2.7794964, upper bound: 2.7794558
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.01
Output dim: 0, lower bound: -2.7794558, upper bound: 2.7794964
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.01
Output dim: 0, lower bound: -2.7770190, upper bound: 2.7770346
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.01
Output dim: 0, lower bound: -2.7770190, upper bound: 2.7770346

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756370, upper bound: 2.7793940
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756370, upper bound: 2.7791934
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754198, upper bound: 2.7791963
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7790509, upper bound: 2.7784898
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7636225, upper bound: 2.7614763
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7636225, upper bound: 2.7620635
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7739194, upper bound: 2.7750868
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752942, upper bound: 2.7736922
time: 0.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.07 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 0, lower bound: -2.7756370, upper bound: 2.7793940
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 0, lower bound: -2.7756370, upper bound: 2.7791934
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 0, lower bound: -2.7754198, upper bound: 2.7791963
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 0, lower bound: -2.7790509, upper bound: 2.7784898
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.07
Output dim: 0, lower bound: -2.7636225, upper bound: 2.7614763
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.07
Output dim: 0, lower bound: -2.7636225, upper bound: 2.7620635
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 0, lower bound: -2.7739194, upper bound: 2.7750868
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.07
Output dim: 0, lower bound: -2.7752942, upper bound: 2.7736922

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756370, upper bound: 2.7793940
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756370, upper bound: 2.7793913
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7792196, upper bound: 2.7786880
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756370, upper bound: 2.7791934
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749874, upper bound: 2.7791958
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754198, upper bound: 2.7789274
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7580089, upper bound: 2.7595082
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7578066, upper bound: 2.7580468
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7566586, upper bound: 2.7566586
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7566586, upper bound: 2.7566586
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
time: 0.39 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.17 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -2.7756370, upper bound: 2.7793940
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -2.7756370, upper bound: 2.7793913
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -2.7792196, upper bound: 2.7786880
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -2.7756370, upper bound: 2.7791934
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -2.7749874, upper bound: 2.7791958
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -2.7754198, upper bound: 2.7789274
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.17
Output dim: 0, lower bound: -2.7580089, upper bound: 2.7595082
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.17
Output dim: 0, lower bound: -2.7578066, upper bound: 2.7580468
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.17
Output dim: 0, lower bound: -2.7566586, upper bound: 2.7566586
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.17
Output dim: 0, lower bound: -2.7566586, upper bound: 2.7566586
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.17
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.17
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7755129
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7793940
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7586014, upper bound: 2.7619741
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7586014, upper bound: 2.7619741
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7611867, upper bound: 2.7624629
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7602871, upper bound: 2.7624629
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7794346, upper bound: 2.7791934
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7790476
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754333, upper bound: 2.7747947
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755404, upper bound: 2.7791958
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754211, upper bound: 2.7747947
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7789274
time: 0.37 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7755129
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7793940
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7586014, upper bound: 2.7619741
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7586014, upper bound: 2.7619741
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7611867, upper bound: 2.7624629
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7602871, upper bound: 2.7624629
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7794346, upper bound: 2.7791934
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7790476
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7754333, upper bound: 2.7747947
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7755404, upper bound: 2.7791958
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7754211, upper bound: 2.7747947
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7789274

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7763946
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7756439
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757080, upper bound: 2.7764157
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7765276, upper bound: 2.7757665
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7786160
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782794
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752518, upper bound: 2.7746755
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753309, upper bound: 2.7746755
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7539397, upper bound: 2.7580261
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7539397, upper bound: 2.7580261
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7747947
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752894, upper bound: 2.7747947
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7537328, upper bound: 2.7558894
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7537328, upper bound: 2.7563438
time: 0.33 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.14 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7763946
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7756439
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 0, lower bound: -2.7757080, upper bound: 2.7764157
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 0, lower bound: -2.7765276, upper bound: 2.7757665
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7786160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782794
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 0, lower bound: -2.7752518, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 0, lower bound: -2.7753309, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.14
Output dim: 0, lower bound: -2.7539397, upper bound: 2.7580261
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.14
Output dim: 0, lower bound: -2.7539397, upper bound: 2.7580261
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7747947
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.14
Output dim: 0, lower bound: -2.7752894, upper bound: 2.7747947
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.14
Output dim: 0, lower bound: -2.7537328, upper bound: 2.7558894
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.14
Output dim: 0, lower bound: -2.7537328, upper bound: 2.7563438

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7755711
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7754417
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746771, upper bound: 2.7755682
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748525, upper bound: 2.7754137
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7754917
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747621
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7578181
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532967, upper bound: 2.7578181
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750833, upper bound: 2.7746755
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746776, upper bound: 2.7746755
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751734, upper bound: 2.7746755
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751748, upper bound: 2.7746755
time: 0.38 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7755711
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7754417
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7746771, upper bound: 2.7755682
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7748525, upper bound: 2.7754137
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7754917
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747621
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7578181
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7532967, upper bound: 2.7578181
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7750833, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7746776, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7751734, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.22
Output dim: 0, lower bound: -2.7751748, upper bound: 2.7746755

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.16
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
Binary search (step 5): status=Status.VERIFIED, low=0.1968750, high=0.2000000, mid=0.1968750, abs_max=3.285133123397827
rel_dist={0: [-2.7804845941221, 2.7804845941221004]}

## Binary search (step 6) starts
Candidate diff: 0.1984375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801671, upper bound: 2.7801671
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801671, upper bound: 2.7801671
time: 0.37 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -2.7801671, upper bound: 2.7801671
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -2.7801671, upper bound: 2.7801671

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7777000, upper bound: 2.7785311
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7785311, upper bound: 2.7777106
time: 0.39 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801380, upper bound: 2.7801151
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800837, upper bound: 2.7801380
time: 0.47 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.17 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -2.7777000, upper bound: 2.7785311
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -2.7785311, upper bound: 2.7777106
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -2.7801380, upper bound: 2.7801151
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -2.7800837, upper bound: 2.7801380

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7777000, upper bound: 2.7784632
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768956, upper bound: 2.7785311
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7566586, upper bound: 2.7566586
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7566586, upper bound: 2.7566586
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801380, upper bound: 2.7771814
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800073, upper bound: 2.7801151
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791934, upper bound: 2.7784977
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7793940, upper bound: 2.7756370
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.11 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -2.7777000, upper bound: 2.7784632
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -2.7768956, upper bound: 2.7785311
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.11
Output dim: 0, lower bound: -2.7566586, upper bound: 2.7566586
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.11
Output dim: 0, lower bound: -2.7566586, upper bound: 2.7566586
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -2.7801380, upper bound: 2.7771814
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -2.7800073, upper bound: 2.7801151
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -2.7791934, upper bound: 2.7784977
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.11
Output dim: 0, lower bound: -2.7793940, upper bound: 2.7756370

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7783191
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7785020
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7621388, upper bound: 2.7577086
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7621388, upper bound: 2.7577086
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800073, upper bound: 2.7801151
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7793418, upper bound: 2.7799277
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757477, upper bound: 2.7762828
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7765427, upper bound: 2.7756370
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757404, upper bound: 2.7756370
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7765427, upper bound: 2.7756370
time: 0.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.09 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7783191
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7785020
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7621388, upper bound: 2.7577086
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7621388, upper bound: 2.7577086
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7800073, upper bound: 2.7801151
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7793418, upper bound: 2.7799277
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7757477, upper bound: 2.7762828
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7765427, upper bound: 2.7756370
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7757404, upper bound: 2.7756370
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.7765427, upper bound: 2.7756370

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516825
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516825
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7763176
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7761160
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791934, upper bound: 2.7794346
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791934, upper bound: 2.7794346
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7792388, upper bound: 2.7799277
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7769265
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757142, upper bound: 2.7755129
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757209, upper bound: 2.7762231
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7765285, upper bound: 2.7755129
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7764715, upper bound: 2.7755297
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757477, upper bound: 2.7756370
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756718, upper bound: 2.7756370
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7765285, upper bound: 2.7755129
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7764715, upper bound: 2.7755129
time: 0.38 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.14 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516825
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516825
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7763176
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7761160
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7791934, upper bound: 2.7794346
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7791934, upper bound: 2.7794346
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7792388, upper bound: 2.7799277
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7769265
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7757142, upper bound: 2.7755129
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7757209, upper bound: 2.7762231
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7765285, upper bound: 2.7755129
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7764715, upper bound: 2.7755297
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7757477, upper bound: 2.7756370
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7756718, upper bound: 2.7756370
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7765285, upper bound: 2.7755129
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.14
Output dim: 0, lower bound: -2.7764715, upper bound: 2.7755129

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7762559
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7763176
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757665, upper bound: 2.7765276
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7764157, upper bound: 2.7757080
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757665, upper bound: 2.7765276
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7764157, upper bound: 2.7757080
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788344
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751457, upper bound: 2.7789023
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746913
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747043
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757142, upper bound: 2.7755129
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7755129
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757142, upper bound: 2.7755129
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757209, upper bound: 2.7755129
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756439, upper bound: 2.7755129
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7755129
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.36 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.20 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7762559
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7763176
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7757665, upper bound: 2.7765276
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7764157, upper bound: 2.7757080
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7757665, upper bound: 2.7765276
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7764157, upper bound: 2.7757080
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788344
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7751457, upper bound: 2.7789023
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746913
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747043
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7757142, upper bound: 2.7755129
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7755129
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7757142, upper bound: 2.7755129
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7757209, upper bound: 2.7755129
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7756439, upper bound: 2.7755129
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7755129
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.20
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7754917
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7754294
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751591, upper bound: 2.7748525
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755682, upper bound: 2.7747278
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748198, upper bound: 2.7756812
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749221, upper bound: 2.7751748
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7756822
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7748544
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535687, upper bound: 2.7558570
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535718, upper bound: 2.7560691
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746913
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746790
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536538
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.36 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.18 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7754917
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7754294
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7751591, upper bound: 2.7748525
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7755682, upper bound: 2.7747278
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7748198, upper bound: 2.7756812
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7749221, upper bound: 2.7751748
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7756822
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7748544
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7535687, upper bound: 2.7558570
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7535718, upper bound: 2.7560691
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746913
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746790
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536538
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.18
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.24
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
Binary search (step 6): status=Status.VERIFIED, low=0.1984375, high=0.2000000, mid=0.1984375, abs_max=3.285133123397827
rel_dist={0: [-2.7804845941221, 2.7804845941221004]}

## Binary search (step 7) starts
Candidate diff: 0.1992187


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804846, upper bound: 2.7803453
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803453, upper bound: 2.7804846
time: 0.37 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -2.7804846, upper bound: 2.7803453
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -2.7803453, upper bound: 2.7804846

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7780947, upper bound: 2.7791573
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7780947
time: 0.37 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803346, upper bound: 2.7803555
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801481, upper bound: 2.7804611
time: 0.37 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.07 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -2.7780947, upper bound: 2.7791573
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7780947
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -2.7803346, upper bound: 2.7803555
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -2.7801481, upper bound: 2.7804611

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7780783, upper bound: 2.7787635
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772449, upper bound: 2.7791573
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791105, upper bound: 2.7780947
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7780859
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801481, upper bound: 2.7803555
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802400, upper bound: 2.7802567
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772449, upper bound: 2.7791573
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7787635, upper bound: 2.7780783
time: 0.35 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.14 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.14
Output dim: 0, lower bound: -2.7780783, upper bound: 2.7787635
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.14
Output dim: 0, lower bound: -2.7772449, upper bound: 2.7791573
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.14
Output dim: 0, lower bound: -2.7791105, upper bound: 2.7780947
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.14
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7780859
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.14
Output dim: 0, lower bound: -2.7801481, upper bound: 2.7803555
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.14
Output dim: 0, lower bound: -2.7802400, upper bound: 2.7802567
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.14
Output dim: 0, lower bound: -2.7772449, upper bound: 2.7791573
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.14
Output dim: 0, lower bound: -2.7787635, upper bound: 2.7780783

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776419, upper bound: 2.7781942
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776495, upper bound: 2.7771264
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751479, upper bound: 2.7757118
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751479, upper bound: 2.7754546
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7790974, upper bound: 2.7771527
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772439, upper bound: 2.7780466
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791401, upper bound: 2.7772216
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788129, upper bound: 2.7780330
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7803401
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7798212
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7802410
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7802298
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772313, upper bound: 2.7788129
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772216, upper bound: 2.7791401
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7787572, upper bound: 2.7776674
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786448, upper bound: 2.7780390
time: 0.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -2.7776419, upper bound: 2.7781942
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -2.7776495, upper bound: 2.7771264
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -2.7751479, upper bound: 2.7757118
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -2.7751479, upper bound: 2.7754546
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -2.7790974, upper bound: 2.7771527
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -2.7772439, upper bound: 2.7780466
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -2.7791401, upper bound: 2.7772216
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -2.7788129, upper bound: 2.7780330
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7803401
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7798212
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7802410
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7802298
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -2.7772313, upper bound: 2.7788129
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -2.7772216, upper bound: 2.7791401
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -2.7787572, upper bound: 2.7776674
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -2.7786448, upper bound: 2.7780390

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769220, upper bound: 2.7768956
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776419, upper bound: 2.7781942
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747971, upper bound: 2.7747947
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748975, upper bound: 2.7747947
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751479, upper bound: 2.7751988
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751426, upper bound: 2.7757118
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751479, upper bound: 2.7749654
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751426, upper bound: 2.7754546
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758667, upper bound: 2.7756882
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758658, upper bound: 2.7756882
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791401, upper bound: 2.7772216
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7784716, upper bound: 2.7771696
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788129, upper bound: 2.7772313
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7780299
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800327
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799838, upper bound: 2.7800336
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771696, upper bound: 2.7784716
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7771527
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800039
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7792388, upper bound: 2.7799277
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7563104, upper bound: 2.7565483
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7563104, upper bound: 2.7576092
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772313, upper bound: 2.7788129
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7787987
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772216, upper bound: 2.7791401
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7790974
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771159, upper bound: 2.7775931
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7781126, upper bound: 2.7775897
time: 0.39 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7769220, upper bound: 2.7768956
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7776419, upper bound: 2.7781942
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7747971, upper bound: 2.7747947
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7748975, upper bound: 2.7747947
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7751479, upper bound: 2.7751988
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7751426, upper bound: 2.7757118
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7751479, upper bound: 2.7749654
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7751426, upper bound: 2.7754546
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7758667, upper bound: 2.7756882
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7758658, upper bound: 2.7756882
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7791401, upper bound: 2.7772216
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7784716, upper bound: 2.7771696
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7788129, upper bound: 2.7772313
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7780299
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800327
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7799838, upper bound: 2.7800336
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7771696, upper bound: 2.7784716
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7771527
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800039
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7792388, upper bound: 2.7799277
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7563104, upper bound: 2.7565483
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7563104, upper bound: 2.7576092
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7772313, upper bound: 2.7788129
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7787987
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7772216, upper bound: 2.7791401
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7790974
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7771159, upper bound: 2.7775931
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.7781126, upper bound: 2.7775897

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769075, upper bound: 2.7768811
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7775897, upper bound: 2.7781126
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772845, upper bound: 2.7781803
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748442, upper bound: 2.7748442
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750250, upper bound: 2.7750669
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748442, upper bound: 2.7753961
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750225, upper bound: 2.7755987
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748902, upper bound: 2.7748442
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750225, upper bound: 2.7748442
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748442, upper bound: 2.7753468
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750225, upper bound: 2.7753451
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516825, upper bound: 2.7516653
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7765285, upper bound: 2.7755129
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7765285, upper bound: 2.7755129
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7476016
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7769212
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759051, upper bound: 2.7749252
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759051, upper bound: 2.7749091
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789702
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789702
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754041, upper bound: 2.7790491
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7781960, upper bound: 2.7790491
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7780020
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769212, upper bound: 2.7770806
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770397, upper bound: 2.7768811
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds
Binary search (step 7): status=Status.UNKNOWN, low=0.1984375, high=0.1992187, mid=0.1992187, abs_max=3.285133123397827
rel_dist={0: [-2.7804845941221, 2.7804845941221004]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.1984374881722033
execution time: 1152.59 seconds
