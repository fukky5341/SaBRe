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
execution time: IAR + LP analysis = 1.40 + 1.03 = 2.44 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2.7804846, upper bound: 2.7804846


# Binary Search by BASE starts (time budget: 1197.56 seconds, max iter: 100)

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
Binary search time: 45.24 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1152.32 seconds

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804502
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804673
time: 0.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.83 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804502
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.83
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
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7803401
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7804290
time: 0.31 seconds

## BFS RS instance: RS_RSZ2

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
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804290, upper bound: 2.7802298
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7804438
time: 0.38 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.20 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7803401
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7804290
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -2.7804290, upper bound: 2.7802298
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7804438

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804438, upper bound: 2.7800772
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7803401
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7772637
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801314, upper bound: 2.7804290
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804290, upper bound: 2.7801314
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7802298
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800772, upper bound: 2.7803173
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800772, upper bound: 2.7804438
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.18 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -2.7804438, upper bound: 2.7800772
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7803401
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7772637
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -2.7801314, upper bound: 2.7804290
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -2.7804290, upper bound: 2.7801314
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7802298
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -2.7800772, upper bound: 2.7803173
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -2.7800772, upper bound: 2.7804438

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803322, upper bound: 2.7771527
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804438, upper bound: 2.7800772
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803173, upper bound: 2.7803401
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801915, upper bound: 2.7802410
time: 0.34 seconds

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
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771057, upper bound: 2.7770397
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799926, upper bound: 2.7768811
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
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801314, upper bound: 2.7804290
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7802437
time: 0.36 seconds

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
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802437, upper bound: 2.7771527
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804290, upper bound: 2.7801314
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7799926
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770397, upper bound: 2.7771057
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802410, upper bound: 2.7801915
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803401, upper bound: 2.7803173
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800772, upper bound: 2.7804438
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7803322
time: 0.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7803322, upper bound: 2.7771527
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7804438, upper bound: 2.7800772
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7803173, upper bound: 2.7803401
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7801915, upper bound: 2.7802410
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7771057, upper bound: 2.7770397
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7799926, upper bound: 2.7768811
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7801314, upper bound: 2.7804290
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7802437
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7802437, upper bound: 2.7771527
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7804290, upper bound: 2.7801314
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7799926
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7770397, upper bound: 2.7771057
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7802410, upper bound: 2.7801915
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7803401, upper bound: 2.7803173
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7800772, upper bound: 2.7804438
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7803322

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800401, upper bound: 2.7768811
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7798526
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7771159
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800327
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799838, upper bound: 2.7800336
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800039
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7792388, upper bound: 2.7799277
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7770397
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799926, upper bound: 2.7768811
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7795280, upper bound: 2.7768811
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800654
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799066, upper bound: 2.7800947
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800065
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7769265
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769265, upper bound: 2.7768811
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800065, upper bound: 2.7768811
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800947, upper bound: 2.7799066
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800654, upper bound: 2.7768811
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7795280
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7799926
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770397, upper bound: 2.7771057
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.36 seconds

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799277, upper bound: 2.7792388
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800039, upper bound: 2.7768811
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800336, upper bound: 2.7799838
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800327, upper bound: 2.7768811
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7801145
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7798526, upper bound: 2.7801145
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800401
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.38 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.34 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7800401, upper bound: 2.7768811
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7798526
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7771159
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800327
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7799838, upper bound: 2.7800336
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800039
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7792388, upper bound: 2.7799277
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7770397
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7799926, upper bound: 2.7768811
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7795280, upper bound: 2.7768811
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800654
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7799066, upper bound: 2.7800947
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800065
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7769265
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7769265, upper bound: 2.7768811
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7800065, upper bound: 2.7768811
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7800947, upper bound: 2.7799066
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7800654, upper bound: 2.7768811
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7795280
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7799926
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7770397, upper bound: 2.7771057
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7799277, upper bound: 2.7792388
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7800039, upper bound: 2.7768811
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7800336, upper bound: 2.7799838
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7800327, upper bound: 2.7768811
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7801145
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7798526, upper bound: 2.7801145
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800401
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.34
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.37 seconds

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752518, upper bound: 2.7746755
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782794, upper bound: 2.7746755
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788146
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7779866, upper bound: 2.7788146
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782361, upper bound: 2.7746755
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789702
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789702
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754041, upper bound: 2.7790491
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7781960, upper bound: 2.7790491
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789675
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789675
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788344
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751457, upper bound: 2.7789023
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7755174
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750874, upper bound: 2.7753317
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770375, upper bound: 2.7746755
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786160, upper bound: 2.7746755
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778045, upper bound: 2.7746755
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7783349
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7753225
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7787676, upper bound: 2.7783621
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7787676, upper bound: 2.7753216
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782929
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7753309
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746913
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747043
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747043, upper bound: 2.7746755
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746913, upper bound: 2.7746755
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753309, upper bound: 2.7746755
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782929, upper bound: 2.7746755
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

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
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753225, upper bound: 2.7746755
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7783349, upper bound: 2.7746755
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7778045
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747076
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7786160
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7770375
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753317, upper bound: 2.7750874
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755174, upper bound: 2.7746755
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789023, upper bound: 2.7751457
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788344, upper bound: 2.7746755
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789675, upper bound: 2.7746755
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789675, upper bound: 2.7746755
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7790491, upper bound: 2.7781960
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7790491, upper bound: 2.7754041
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789702, upper bound: 2.7746755
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789702, upper bound: 2.7746755
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782361
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788146, upper bound: 2.7779866
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788146, upper bound: 2.7746755
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782794
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7752518
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.42 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7752518, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7782794, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788146
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7779866, upper bound: 2.7788146
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7782361, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789702
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789702
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7754041, upper bound: 2.7790491
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7781960, upper bound: 2.7790491
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789675
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789675
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788344
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7751457, upper bound: 2.7789023
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7755174
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7750874, upper bound: 2.7753317
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7770375, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7786160, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7778045, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7783349
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7753225
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7787676, upper bound: 2.7783621
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7787676, upper bound: 2.7753216
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782929
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7753309
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746913
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747043
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7747043, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746913, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7753309, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7782929, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7753216, upper bound: 2.7787676
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7783621, upper bound: 2.7787676
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7753225, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7783349, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7778045
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747076
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7786160
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7770375
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7753317, upper bound: 2.7750874
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7755174, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7789023, upper bound: 2.7751457
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7788344, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7789675, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7789675, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7790491, upper bound: 2.7781960
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7790491, upper bound: 2.7754041
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7789702, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7789702, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782361
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7788146, upper bound: 2.7779866
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7788146, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782794
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7752518
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for candidate selection: 0.12 seconds

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
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535127, upper bound: 2.7533155
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533197
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533820
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533978
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7542692, upper bound: 2.7533778
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7560946, upper bound: 2.7533778
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533820
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532991, upper bound: 2.7533838
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582459, upper bound: 2.7533354
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582459, upper bound: 2.7533354
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7579100
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7579100
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535131, upper bound: 2.7563966
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535214, upper bound: 2.7564464
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7534652, upper bound: 2.7577225
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7534652, upper bound: 2.7577225
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7549012, upper bound: 2.7564821
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7549012, upper bound: 2.7564821
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7575138
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7575679
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535250, upper bound: 2.7562011
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535480, upper bound: 2.7563167
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7555630
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7560268
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535687, upper bound: 2.7558570
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535718, upper bound: 2.7560691
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
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535729
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535729
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536209
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536209
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533188, upper bound: 2.7534824
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533188, upper bound: 2.7534824
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535024, upper bound: 2.7535666
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535024, upper bound: 2.7535666
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7565964, upper bound: 2.7532850
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7545022, upper bound: 2.7532625
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533225, upper bound: 2.7535629
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533225, upper bound: 2.7535629
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7562925, upper bound: 2.7533260
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7548464, upper bound: 2.7533172
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7582293
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7582293
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7539970
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7575015
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532883, upper bound: 2.7566529
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533996, upper bound: 2.7539491
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7577203
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7577203
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541317
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541317
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541962
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541962
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536538
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7536538, upper bound: 2.7532625
time: 0.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7541962, upper bound: 2.7532625
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535629, upper bound: 2.7532625
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7541317, upper bound: 2.7532625
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7541317, upper bound: 2.7532625
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577203, upper bound: 2.7532625
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577203, upper bound: 2.7532625
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7539491, upper bound: 2.7533996
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7566529, upper bound: 2.7532883
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7575015, upper bound: 2.7533279
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7539970, upper bound: 2.7532625
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582293, upper bound: 2.7532625
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582293, upper bound: 2.7532625
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7548464
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533260, upper bound: 2.7562925
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535629, upper bound: 2.7533225
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535629, upper bound: 2.7533225
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7545022
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532850, upper bound: 2.7565964
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535666, upper bound: 2.7535024
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535666, upper bound: 2.7535024
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7534824, upper bound: 2.7533188
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7534824, upper bound: 2.7533188
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7536209, upper bound: 2.7532625
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7536209, upper bound: 2.7532625
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535729, upper bound: 2.7532625
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535729, upper bound: 2.7532625
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7560691, upper bound: 2.7535718
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7558570, upper bound: 2.7535687
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7560268, upper bound: 2.7532625
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7555630, upper bound: 2.7532625
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7563167, upper bound: 2.7535480
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7562011, upper bound: 2.7535250
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7575679, upper bound: 2.7532625
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7575138, upper bound: 2.7532625
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7564821, upper bound: 2.7549012
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7564821, upper bound: 2.7549012
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577225, upper bound: 2.7534652
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577225, upper bound: 2.7534652
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7564464, upper bound: 2.7535214
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7563966, upper bound: 2.7535131
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7579100, upper bound: 2.7532625
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7579100, upper bound: 2.7532625
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533354, upper bound: 2.7582459
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533354, upper bound: 2.7582459
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532991
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532625
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533778, upper bound: 2.7560946
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533778, upper bound: 2.7542692
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533778, upper bound: 2.7532625
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532625
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532967, upper bound: 2.7578181
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532967, upper bound: 2.7578181
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532967, upper bound: 2.7535127
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535127
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.36 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.67 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7535127, upper bound: 2.7533155
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533197
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533820
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533978
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7542692, upper bound: 2.7533778
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7560946, upper bound: 2.7533778
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533820
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532991, upper bound: 2.7533838
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7582459, upper bound: 2.7533354
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7582459, upper bound: 2.7533354
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7579100
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7579100
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7535131, upper bound: 2.7563966
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7535214, upper bound: 2.7564464
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7534652, upper bound: 2.7577225
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7534652, upper bound: 2.7577225
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7549012, upper bound: 2.7564821
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7549012, upper bound: 2.7564821
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7575138
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7575679
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7535250, upper bound: 2.7562011
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7535480, upper bound: 2.7563167
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7555630
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7560268
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7535687, upper bound: 2.7558570
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7535718, upper bound: 2.7560691
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535729
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535729
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536209
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536209
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7533188, upper bound: 2.7534824
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7533188, upper bound: 2.7534824
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7535024, upper bound: 2.7535666
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7535024, upper bound: 2.7535666
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7565964, upper bound: 2.7532850
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7545022, upper bound: 2.7532625
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7533225, upper bound: 2.7535629
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7533225, upper bound: 2.7535629
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7562925, upper bound: 2.7533260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7548464, upper bound: 2.7533172
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7582293
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7582293
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7539970
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7575015
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532883, upper bound: 2.7566529
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7533996, upper bound: 2.7539491
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7577203
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7577203
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541317
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541317
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541962
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541962
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536538
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7536538, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7541962, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7535629, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7541317, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7541317, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7577203, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7577203, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7539491, upper bound: 2.7533996
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7566529, upper bound: 2.7532883
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7575015, upper bound: 2.7533279
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7539970, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7582293, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7582293, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7548464
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7533260, upper bound: 2.7562925
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7535629, upper bound: 2.7533225
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7535629, upper bound: 2.7533225
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7545022
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532850, upper bound: 2.7565964
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7535666, upper bound: 2.7535024
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7535666, upper bound: 2.7535024
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7534824, upper bound: 2.7533188
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7534824, upper bound: 2.7533188
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7536209, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7536209, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7535729, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7535729, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7560691, upper bound: 2.7535718
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7558570, upper bound: 2.7535687
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7560268, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7555630, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7563167, upper bound: 2.7535480
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7562011, upper bound: 2.7535250
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7575679, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7575138, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7564821, upper bound: 2.7549012
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7564821, upper bound: 2.7549012
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7577225, upper bound: 2.7534652
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7577225, upper bound: 2.7534652
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7564464, upper bound: 2.7535214
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7563966, upper bound: 2.7535131
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7579100, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7579100, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7533354, upper bound: 2.7582459
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7533354, upper bound: 2.7582459
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532991
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7533778, upper bound: 2.7560946
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7533778, upper bound: 2.7542692
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7533778, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532967, upper bound: 2.7578181
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532967, upper bound: 2.7578181
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532967, upper bound: 2.7535127
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535127
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.67
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
Binary search (step 0): status=Status.VERIFIED, low=0.1000000, high=0.2000000, mid=0.1000000, abs_max=3.285133123397827
rel_dist={0: [-2.7804845941221, 2.7804845941221004]}

## Binary search (step 1) starts
Candidate diff: 0.1500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

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
- Time for RS candidates: 0.86 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.86
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804502
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.86
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
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7803401
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7804290
time: 0.37 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7802298
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7804438
time: 0.37 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.19 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7803401
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7804290
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7802298
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7804438

## BFS RS instance: RS_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7803401
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7798212, upper bound: 2.7802410
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7772637
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801314, upper bound: 2.7804290
time: 0.37 seconds

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
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804290, upper bound: 2.7801314
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7802298
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802410, upper bound: 2.7804438
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7798212, upper bound: 2.7803322
time: 0.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.26 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.26
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7803401
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.26
Output dim: 0, lower bound: -2.7798212, upper bound: 2.7802410
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.26
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7772637
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.26
Output dim: 0, lower bound: -2.7801314, upper bound: 2.7804290
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.26
Output dim: 0, lower bound: -2.7804290, upper bound: 2.7801314
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.26
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7802298
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.26
Output dim: 0, lower bound: -2.7802410, upper bound: 2.7804438
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.26
Output dim: 0, lower bound: -2.7798212, upper bound: 2.7803322

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803322, upper bound: 2.7771527
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7803401
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
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804438, upper bound: 2.7800772
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7802410
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7771527
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7798212, upper bound: 2.7772637
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801314, upper bound: 2.7804290
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7802437
time: 0.35 seconds

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
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802437, upper bound: 2.7771527
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804290, upper bound: 2.7801314
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
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7798212
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7802298
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802410, upper bound: 2.7801915
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800772, upper bound: 2.7804438
time: 0.33 seconds

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
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803401, upper bound: 2.7803173
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7803322
time: 0.34 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7803322, upper bound: 2.7771527
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7803401
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7804438, upper bound: 2.7800772
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7802410
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7771527
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7798212, upper bound: 2.7772637
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7801314, upper bound: 2.7804290
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7802437
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7802437, upper bound: 2.7771527
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7804290, upper bound: 2.7801314
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7798212
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7802298
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7802410, upper bound: 2.7801915
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7800772, upper bound: 2.7804438
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7803401, upper bound: 2.7803173
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7803322

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800401, upper bound: 2.7768811
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800327
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799838, upper bound: 2.7800336
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
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7798526
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7771159
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800039
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7792388, upper bound: 2.7799277
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799926, upper bound: 2.7768811
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7770397
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7795280, upper bound: 2.7768811
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800654
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799066, upper bound: 2.7800947
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800065
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7769265
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769265, upper bound: 2.7768811
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800065, upper bound: 2.7768811
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800947, upper bound: 2.7799066
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800654, upper bound: 2.7768811
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7795280
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770397, upper bound: 2.7771057
time: 0.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7799926
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799277, upper bound: 2.7792388
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800039, upper bound: 2.7768811
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771159, upper bound: 2.7801145
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7798526, upper bound: 2.7801145
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800336, upper bound: 2.7799838
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800327, upper bound: 2.7768811
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800401
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.34 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.36 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7800401, upper bound: 2.7768811
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800327
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7799838, upper bound: 2.7800336
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7798526
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7771159
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800039
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7792388, upper bound: 2.7799277
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7799926, upper bound: 2.7768811
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7770397
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7795280, upper bound: 2.7768811
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800654
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7799066, upper bound: 2.7800947
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800065
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7769265
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7769265, upper bound: 2.7768811
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7800065, upper bound: 2.7768811
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7800947, upper bound: 2.7799066
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7800654, upper bound: 2.7768811
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7795280
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7770397, upper bound: 2.7771057
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7799926
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7799277, upper bound: 2.7792388
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7800039, upper bound: 2.7768811
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7771159, upper bound: 2.7801145
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7798526, upper bound: 2.7801145
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7800336, upper bound: 2.7799838
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7800327, upper bound: 2.7768811
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800401
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752518, upper bound: 2.7746755
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782794, upper bound: 2.7746755
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789702
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789702
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754041, upper bound: 2.7790491
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7781960, upper bound: 2.7790491
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788146
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7779866, upper bound: 2.7788146
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782361, upper bound: 2.7746755
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789675
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789675
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788344
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751457, upper bound: 2.7789023
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770375, upper bound: 2.7746755
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786160, upper bound: 2.7746755
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7755174
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750874, upper bound: 2.7753317
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747076, upper bound: 2.7746755
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778045, upper bound: 2.7746755
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7783349
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7753225
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7787676, upper bound: 2.7783621
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7787676, upper bound: 2.7753216
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782929
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7753309
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746913
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747043
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747043, upper bound: 2.7746755
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746913, upper bound: 2.7746755
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753309, upper bound: 2.7746755
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782929, upper bound: 2.7746755
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753216, upper bound: 2.7787676
time: 0.38 seconds

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753225, upper bound: 2.7746755
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7783349, upper bound: 2.7746755
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7778045
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747076
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753317, upper bound: 2.7750874
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755174, upper bound: 2.7746755
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7786160
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7770375
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789023, upper bound: 2.7751457
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788344, upper bound: 2.7746755
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789675, upper bound: 2.7746755
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789675, upper bound: 2.7746755
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782361
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788146, upper bound: 2.7779866
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788146, upper bound: 2.7746755
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7790491, upper bound: 2.7781960
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7790491, upper bound: 2.7754041
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789702, upper bound: 2.7746755
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789702, upper bound: 2.7746755
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782794
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7752518
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.36 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7752518, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7782794, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789702
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789702
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7754041, upper bound: 2.7790491
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7781960, upper bound: 2.7790491
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788146
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7779866, upper bound: 2.7788146
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7782361, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789675
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789675
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788344
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7751457, upper bound: 2.7789023
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7770375, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7786160, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7755174
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7750874, upper bound: 2.7753317
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7747076, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7778045, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7783349
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7753225
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7787676, upper bound: 2.7783621
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7787676, upper bound: 2.7753216
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782929
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7753309
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746913
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747043
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7747043, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746913, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7753309, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7782929, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7753216, upper bound: 2.7787676
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7783621, upper bound: 2.7787676
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7753225, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7783349, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7778045
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747076
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7753317, upper bound: 2.7750874
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7755174, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7786160
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7770375
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7789023, upper bound: 2.7751457
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7788344, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7789675, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7789675, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782361
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7788146, upper bound: 2.7779866
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7788146, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7790491, upper bound: 2.7781960
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7790491, upper bound: 2.7754041
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7789702, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7789702, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782794
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7752518
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for candidate selection: 0.13 seconds

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
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535127, upper bound: 2.7533155
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535127, upper bound: 2.7533197
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7579100
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7579100
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535131, upper bound: 2.7563966
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535214, upper bound: 2.7564464
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7534652, upper bound: 2.7577225
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7534652, upper bound: 2.7577225
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7549012, upper bound: 2.7564821
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7549012, upper bound: 2.7564821
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533820
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533978
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7542692, upper bound: 2.7533778
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7560946, upper bound: 2.7533778
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533820
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532991, upper bound: 2.7533838
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582459, upper bound: 2.7533354
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582459, upper bound: 2.7533354
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7575138
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7575679
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535250, upper bound: 2.7562011
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535480, upper bound: 2.7563167
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7555630
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7560268
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535687, upper bound: 2.7558570
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535718, upper bound: 2.7560691
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535729
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535729
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535024, upper bound: 2.7535666
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535024, upper bound: 2.7535666
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7565964, upper bound: 2.7532850
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7545022, upper bound: 2.7532625
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536209
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536209
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533188, upper bound: 2.7534824
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533188, upper bound: 2.7534824
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533225, upper bound: 2.7535629
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533225, upper bound: 2.7535629
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7562925, upper bound: 2.7533260
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7548464, upper bound: 2.7533172
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7582293
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7582293
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7539970
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533279, upper bound: 2.7575015
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7566529
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533996, upper bound: 2.7539491
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7577203
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7577203
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541317
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541317
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541962
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541962
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536538
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7536538, upper bound: 2.7532625
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7541962, upper bound: 2.7532625
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7541962, upper bound: 2.7532625
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7541317, upper bound: 2.7532625
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7541317, upper bound: 2.7532625
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577203, upper bound: 2.7532625
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577203, upper bound: 2.7532625
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7539491, upper bound: 2.7533996
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7566529, upper bound: 2.7532883
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7575015, upper bound: 2.7533279
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7539970, upper bound: 2.7532625
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582293, upper bound: 2.7532625
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582293, upper bound: 2.7532625
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533172, upper bound: 2.7548464
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533260, upper bound: 2.7562925
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535629, upper bound: 2.7533225
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535629, upper bound: 2.7533225
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7534824, upper bound: 2.7533188
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7534824, upper bound: 2.7533188
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7536209, upper bound: 2.7532625
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7536209, upper bound: 2.7532625
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7545022
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532850, upper bound: 2.7565964
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535666, upper bound: 2.7535024
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535666, upper bound: 2.7535024
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535729, upper bound: 2.7532625
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535729, upper bound: 2.7532625
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7560691, upper bound: 2.7535718
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7558570, upper bound: 2.7535687
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7560268, upper bound: 2.7532625
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7555630, upper bound: 2.7532625
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7563167, upper bound: 2.7535480
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7562011, upper bound: 2.7535250
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7575679, upper bound: 2.7532625
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7575138, upper bound: 2.7532625
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533354, upper bound: 2.7582459
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533354, upper bound: 2.7582459
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532991
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532625
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533778, upper bound: 2.7560946
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533778, upper bound: 2.7542692
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533978, upper bound: 2.7532625
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532625
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7564821, upper bound: 2.7549012
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7564821, upper bound: 2.7549012
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577225, upper bound: 2.7534652
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577225, upper bound: 2.7534652
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7564464, upper bound: 2.7535214
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7563966, upper bound: 2.7535131
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7579100, upper bound: 2.7532625
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7579100, upper bound: 2.7532625
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532967, upper bound: 2.7578181
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532967, upper bound: 2.7578181
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533197, upper bound: 2.7535127
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533155, upper bound: 2.7535127
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.43 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7535127, upper bound: 2.7533155
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7535127, upper bound: 2.7533197
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7579100
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7579100
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7535131, upper bound: 2.7563966
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7535214, upper bound: 2.7564464
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7534652, upper bound: 2.7577225
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7534652, upper bound: 2.7577225
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7549012, upper bound: 2.7564821
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7549012, upper bound: 2.7564821
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533820
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533978
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7542692, upper bound: 2.7533778
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7560946, upper bound: 2.7533778
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533820
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532991, upper bound: 2.7533838
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7582459, upper bound: 2.7533354
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7582459, upper bound: 2.7533354
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7575138
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7575679
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7535250, upper bound: 2.7562011
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7535480, upper bound: 2.7563167
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7555630
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7560268
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7535687, upper bound: 2.7558570
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7535718, upper bound: 2.7560691
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535729
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535729
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7535024, upper bound: 2.7535666
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7535024, upper bound: 2.7535666
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7565964, upper bound: 2.7532850
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7545022, upper bound: 2.7532625
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536209
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536209
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7533188, upper bound: 2.7534824
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7533188, upper bound: 2.7534824
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7533225, upper bound: 2.7535629
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7533225, upper bound: 2.7535629
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7562925, upper bound: 2.7533260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7548464, upper bound: 2.7533172
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7582293
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7582293
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7539970
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7533279, upper bound: 2.7575015
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7566529
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7533996, upper bound: 2.7539491
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7577203
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7577203
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541317
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541317
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541962
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541962
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536538
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7536538, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7541962, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7541962, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7541317, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7541317, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7577203, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7577203, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7539491, upper bound: 2.7533996
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7566529, upper bound: 2.7532883
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7575015, upper bound: 2.7533279
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7539970, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7582293, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7582293, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7533172, upper bound: 2.7548464
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7533260, upper bound: 2.7562925
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7535629, upper bound: 2.7533225
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7535629, upper bound: 2.7533225
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7534824, upper bound: 2.7533188
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7534824, upper bound: 2.7533188
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7536209, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7536209, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7545022
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532850, upper bound: 2.7565964
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7535666, upper bound: 2.7535024
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7535666, upper bound: 2.7535024
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7535729, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7535729, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7560691, upper bound: 2.7535718
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7558570, upper bound: 2.7535687
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7560268, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7555630, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7563167, upper bound: 2.7535480
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7562011, upper bound: 2.7535250
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7575679, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7575138, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7533354, upper bound: 2.7582459
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7533354, upper bound: 2.7582459
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532991
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7533778, upper bound: 2.7560946
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7533778, upper bound: 2.7542692
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7533978, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7564821, upper bound: 2.7549012
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7564821, upper bound: 2.7549012
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7577225, upper bound: 2.7534652
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7577225, upper bound: 2.7534652
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7564464, upper bound: 2.7535214
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7563966, upper bound: 2.7535131
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7579100, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7579100, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532967, upper bound: 2.7578181
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532967, upper bound: 2.7578181
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7533197, upper bound: 2.7535127
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7533155, upper bound: 2.7535127
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
Binary search (step 1): status=Status.VERIFIED, low=0.1500000, high=0.2000000, mid=0.1500000, abs_max=3.285133123397827
rel_dist={0: [-2.7804845941221, 2.7804845941221004]}

## Binary search (step 2) starts
Candidate diff: 0.1750000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804673, upper bound: 2.7804502
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804673
time: 0.35 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.84 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.84
Output dim: 0, lower bound: -2.7804673, upper bound: 2.7804502
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.84
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804673

## BFS RS instance: RS_RSZ1

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
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7803401
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7804290
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804290, upper bound: 2.7802298
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803401, upper bound: 2.7804438
time: 0.33 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.18 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7803401
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7804290
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -2.7804290, upper bound: 2.7802298
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -2.7803401, upper bound: 2.7804438

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7803401
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7798212, upper bound: 2.7802410
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7772637
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801314, upper bound: 2.7804290
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804290, upper bound: 2.7801314
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7802298
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802410, upper bound: 2.7804438
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7798212, upper bound: 2.7803322
time: 0.35 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.22 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7803401
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -2.7798212, upper bound: 2.7802410
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7772637
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -2.7801314, upper bound: 2.7804290
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -2.7804290, upper bound: 2.7801314
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7802298
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -2.7802410, upper bound: 2.7804438
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -2.7798212, upper bound: 2.7803322

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803322, upper bound: 2.7771527
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803173, upper bound: 2.7803401
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804438, upper bound: 2.7800772
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801915, upper bound: 2.7802410
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
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7771527
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7798212, upper bound: 2.7772637
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801314, upper bound: 2.7804290
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7802437
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802437, upper bound: 2.7771527
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804290, upper bound: 2.7801314
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7798212
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7802298
time: 0.34 seconds

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
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802410, upper bound: 2.7801915
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800772, upper bound: 2.7804438
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803401, upper bound: 2.7803173
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7803322
time: 0.34 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7803322, upper bound: 2.7771527
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7803173, upper bound: 2.7803401
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7804438, upper bound: 2.7800772
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7801915, upper bound: 2.7802410
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7771527
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7798212, upper bound: 2.7772637
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7801314, upper bound: 2.7804290
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7802437
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7802437, upper bound: 2.7771527
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7804290, upper bound: 2.7801314
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7798212
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7802298
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7802410, upper bound: 2.7801915
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7800772, upper bound: 2.7804438
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7803401, upper bound: 2.7803173
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7803322

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800401, upper bound: 2.7768811
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800327
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799838, upper bound: 2.7800336
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7798526
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7771159
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800039
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7792388, upper bound: 2.7799277
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799926, upper bound: 2.7768811
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771057, upper bound: 2.7770397
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7795280, upper bound: 2.7768811
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800654
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799066, upper bound: 2.7800947
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800065
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7769265
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769265, upper bound: 2.7768811
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800065, upper bound: 2.7768811
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800947, upper bound: 2.7799066
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800654, upper bound: 2.7768811
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7795280
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770397, upper bound: 2.7771057
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7799926
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799277, upper bound: 2.7792388
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800039, upper bound: 2.7768811
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7801145
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7798526, upper bound: 2.7801145
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800336, upper bound: 2.7799838
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800327, upper bound: 2.7768811
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800401
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.33 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7800401, upper bound: 2.7768811
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800327
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7799838, upper bound: 2.7800336
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7798526
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7771159
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800039
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7792388, upper bound: 2.7799277
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7799926, upper bound: 2.7768811
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7771057, upper bound: 2.7770397
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7795280, upper bound: 2.7768811
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800654
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7799066, upper bound: 2.7800947
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800065
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7769265
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7769265, upper bound: 2.7768811
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7800065, upper bound: 2.7768811
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7800947, upper bound: 2.7799066
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7800654, upper bound: 2.7768811
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7795280
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7770397, upper bound: 2.7771057
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7799926
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7799277, upper bound: 2.7792388
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7800039, upper bound: 2.7768811
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7801145
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7798526, upper bound: 2.7801145
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7800336, upper bound: 2.7799838
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7800327, upper bound: 2.7768811
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800401
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752518, upper bound: 2.7746755
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782794, upper bound: 2.7746755
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789702
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789702
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754041, upper bound: 2.7790491
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7781960, upper bound: 2.7790491
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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788146
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7779866, upper bound: 2.7788146
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782361, upper bound: 2.7746755
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789675
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789675
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788344
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751457, upper bound: 2.7789023
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770375, upper bound: 2.7746755
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786160, upper bound: 2.7746755
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7755174
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750874, upper bound: 2.7753317
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778045, upper bound: 2.7746755
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7783349
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7753225
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7787676, upper bound: 2.7783621
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7787676, upper bound: 2.7753216
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782929
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7753309
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746913
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747043
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747043, upper bound: 2.7746755
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746913, upper bound: 2.7746755
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753309, upper bound: 2.7746755
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782929, upper bound: 2.7746755
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753216, upper bound: 2.7787676
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7783621, upper bound: 2.7787676
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753225, upper bound: 2.7746755
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7783349, upper bound: 2.7746755
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7778045
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747076
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753317, upper bound: 2.7750874
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755174, upper bound: 2.7746755
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7786160
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7770375
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789023, upper bound: 2.7751457
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788344, upper bound: 2.7746755
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789675, upper bound: 2.7746755
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789675, upper bound: 2.7746755
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782361
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788146, upper bound: 2.7779866
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788146, upper bound: 2.7746755
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7790491, upper bound: 2.7781960
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7790491, upper bound: 2.7754041
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789702, upper bound: 2.7746755
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789702, upper bound: 2.7746755
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782794
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7752518
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.37 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7752518, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7782794, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789702
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789702
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7754041, upper bound: 2.7790491
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7781960, upper bound: 2.7790491
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788146
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7779866, upper bound: 2.7788146
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7782361, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789675
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789675
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788344
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7751457, upper bound: 2.7789023
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7770375, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7786160, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7755174
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7750874, upper bound: 2.7753317
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7778045, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7783349
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7753225
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7787676, upper bound: 2.7783621
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7787676, upper bound: 2.7753216
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782929
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7753309
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746913
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747043
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7747043, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746913, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7753309, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7782929, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7753216, upper bound: 2.7787676
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7783621, upper bound: 2.7787676
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7753225, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7783349, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7778045
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747076
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7753317, upper bound: 2.7750874
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7755174, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7786160
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7770375
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7789023, upper bound: 2.7751457
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7788344, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7789675, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7789675, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782361
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7788146, upper bound: 2.7779866
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7788146, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7790491, upper bound: 2.7781960
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7790491, upper bound: 2.7754041
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7789702, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7789702, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782794
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7752518
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535127, upper bound: 2.7533155
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535127, upper bound: 2.7533197
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7579100
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7579100
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535131, upper bound: 2.7563966
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535214, upper bound: 2.7564464
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7534652, upper bound: 2.7577225
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7534652, upper bound: 2.7577225
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7549012, upper bound: 2.7564821
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7549012, upper bound: 2.7564821
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533820
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533978
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7542692, upper bound: 2.7533778
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7560946, upper bound: 2.7533778
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533820
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532991, upper bound: 2.7533838
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582459, upper bound: 2.7533354
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582459, upper bound: 2.7533354
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7575138
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7575679
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535250, upper bound: 2.7562011
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535480, upper bound: 2.7563167
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7555630
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7560268
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535687, upper bound: 2.7558570
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535718, upper bound: 2.7560691
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535729
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535729
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535024, upper bound: 2.7535666
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535024, upper bound: 2.7535666
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7565964, upper bound: 2.7532850
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7545022, upper bound: 2.7532625
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536209
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536209
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533188, upper bound: 2.7534824
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533188, upper bound: 2.7534824
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533225, upper bound: 2.7535629
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533225, upper bound: 2.7535629
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7562925, upper bound: 2.7533260
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7548464, upper bound: 2.7533172
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7582293
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7582293
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7539970
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533279, upper bound: 2.7575015
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532883, upper bound: 2.7566529
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533996, upper bound: 2.7539491
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7577203
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7577203
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541317
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541317
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541962
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541962
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536538
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7536538, upper bound: 2.7532625
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7541962, upper bound: 2.7532625
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7541962, upper bound: 2.7532625
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7541317, upper bound: 2.7532625
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7541317, upper bound: 2.7532625
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577203, upper bound: 2.7532625
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577203, upper bound: 2.7532625
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7539491, upper bound: 2.7533996
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7566529, upper bound: 2.7532883
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7575015, upper bound: 2.7533279
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7539970, upper bound: 2.7532625
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582293, upper bound: 2.7532625
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582293, upper bound: 2.7532625
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533172, upper bound: 2.7548464
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533260, upper bound: 2.7562925
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535629, upper bound: 2.7533225
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535629, upper bound: 2.7533225
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7534824, upper bound: 2.7533188
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7534824, upper bound: 2.7533188
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7536209, upper bound: 2.7532625
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7536209, upper bound: 2.7532625
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7545022
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532850, upper bound: 2.7565964
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535666, upper bound: 2.7535024
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535666, upper bound: 2.7535024
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535729, upper bound: 2.7532625
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535729, upper bound: 2.7532625
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7560691, upper bound: 2.7535718
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7558570, upper bound: 2.7535687
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7560268, upper bound: 2.7532625
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7555630, upper bound: 2.7532625
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7563167, upper bound: 2.7535480
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7562011, upper bound: 2.7535250
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7575679, upper bound: 2.7532625
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7575138, upper bound: 2.7532625
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533354, upper bound: 2.7582459
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533354, upper bound: 2.7582459
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532991
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533778, upper bound: 2.7560946
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533778, upper bound: 2.7542692
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533978, upper bound: 2.7532625
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532625
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7564821, upper bound: 2.7549012
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7564821, upper bound: 2.7549012
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577225, upper bound: 2.7534652
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577225, upper bound: 2.7534652
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7564464, upper bound: 2.7535214
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7563966, upper bound: 2.7535131
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7579100, upper bound: 2.7532625
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7579100, upper bound: 2.7532625
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532967, upper bound: 2.7578181
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532967, upper bound: 2.7578181
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533197, upper bound: 2.7535127
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533155, upper bound: 2.7535127
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.39 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.62 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7535127, upper bound: 2.7533155
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7535127, upper bound: 2.7533197
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7579100
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7579100
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7535131, upper bound: 2.7563966
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7535214, upper bound: 2.7564464
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7534652, upper bound: 2.7577225
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7534652, upper bound: 2.7577225
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7549012, upper bound: 2.7564821
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7549012, upper bound: 2.7564821
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533820
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533978
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7542692, upper bound: 2.7533778
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7560946, upper bound: 2.7533778
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533820
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532991, upper bound: 2.7533838
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7582459, upper bound: 2.7533354
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7582459, upper bound: 2.7533354
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7575138
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7575679
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7535250, upper bound: 2.7562011
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7535480, upper bound: 2.7563167
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7555630
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7560268
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7535687, upper bound: 2.7558570
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7535718, upper bound: 2.7560691
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535729
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535729
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7535024, upper bound: 2.7535666
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7535024, upper bound: 2.7535666
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7565964, upper bound: 2.7532850
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7545022, upper bound: 2.7532625
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536209
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536209
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7533188, upper bound: 2.7534824
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7533188, upper bound: 2.7534824
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7533225, upper bound: 2.7535629
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7533225, upper bound: 2.7535629
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7562925, upper bound: 2.7533260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7548464, upper bound: 2.7533172
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7582293
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7582293
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7539970
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7533279, upper bound: 2.7575015
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532883, upper bound: 2.7566529
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7533996, upper bound: 2.7539491
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7577203
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7577203
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541317
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541317
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541962
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541962
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536538
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7536538, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7541962, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7541962, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7541317, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7541317, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7577203, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7577203, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7539491, upper bound: 2.7533996
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7566529, upper bound: 2.7532883
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7575015, upper bound: 2.7533279
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7539970, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7582293, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7582293, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7533172, upper bound: 2.7548464
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7533260, upper bound: 2.7562925
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7535629, upper bound: 2.7533225
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7535629, upper bound: 2.7533225
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7534824, upper bound: 2.7533188
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7534824, upper bound: 2.7533188
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7536209, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7536209, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7545022
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532850, upper bound: 2.7565964
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7535666, upper bound: 2.7535024
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7535666, upper bound: 2.7535024
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7535729, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7535729, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7560691, upper bound: 2.7535718
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7558570, upper bound: 2.7535687
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7560268, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7555630, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7563167, upper bound: 2.7535480
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7562011, upper bound: 2.7535250
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7575679, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7575138, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7533354, upper bound: 2.7582459
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7533354, upper bound: 2.7582459
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532991
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7533778, upper bound: 2.7560946
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7533778, upper bound: 2.7542692
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7533978, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7564821, upper bound: 2.7549012
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7564821, upper bound: 2.7549012
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7577225, upper bound: 2.7534652
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7577225, upper bound: 2.7534652
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7564464, upper bound: 2.7535214
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7563966, upper bound: 2.7535131
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7579100, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7579100, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532967, upper bound: 2.7578181
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532967, upper bound: 2.7578181
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7533197, upper bound: 2.7535127
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7533155, upper bound: 2.7535127
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.62
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
Binary search (step 2): status=Status.VERIFIED, low=0.1750000, high=0.2000000, mid=0.1750000, abs_max=3.285133123397827
rel_dist={0: [-2.7804845941221, 2.7804845941221004]}

## Binary search (step 3) starts
Candidate diff: 0.1875000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804502
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804673
time: 0.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.84 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.84
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804502
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.84
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804673

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7802298
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7804438
time: 0.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.26 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7803401
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7804290
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7802298
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7804438

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7803401
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804438, upper bound: 2.7802410
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
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7772637
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801314, upper bound: 2.7804290
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804290, upper bound: 2.7801314
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7802298
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802410, upper bound: 2.7804438
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803401, upper bound: 2.7803322
time: 0.36 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.17 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.17
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7803401
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.17
Output dim: 0, lower bound: -2.7804438, upper bound: 2.7802410
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.17
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7772637
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.17
Output dim: 0, lower bound: -2.7801314, upper bound: 2.7804290
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.17
Output dim: 0, lower bound: -2.7804290, upper bound: 2.7801314
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.17
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7802298
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.17
Output dim: 0, lower bound: -2.7802410, upper bound: 2.7804438
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.17
Output dim: 0, lower bound: -2.7803401, upper bound: 2.7803322

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803322, upper bound: 2.7771527
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7803401
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804438, upper bound: 2.7800772
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801915, upper bound: 2.7802410
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7771527
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7798212, upper bound: 2.7772637
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801314, upper bound: 2.7804290
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7802437
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802437, upper bound: 2.7771527
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804290, upper bound: 2.7801314
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7798212
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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802410, upper bound: 2.7801915
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800772, upper bound: 2.7804438
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803401, upper bound: 2.7803173
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7803322
time: 0.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7803322, upper bound: 2.7771527
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7803401
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7804438, upper bound: 2.7800772
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7801915, upper bound: 2.7802410
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7771527
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7798212, upper bound: 2.7772637
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7801314, upper bound: 2.7804290
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7802437
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7802437, upper bound: 2.7771527
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7804290, upper bound: 2.7801314
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7798212
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7802298
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7802410, upper bound: 2.7801915
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7800772, upper bound: 2.7804438
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7803401, upper bound: 2.7803173
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7771527, upper bound: 2.7803322

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800401, upper bound: 2.7768811
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800327
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799838, upper bound: 2.7800336
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
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7798526
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7771159
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800039
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7792388, upper bound: 2.7799277
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799926, upper bound: 2.7768811
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771057, upper bound: 2.7770397
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7795280, upper bound: 2.7768811
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800654
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799066, upper bound: 2.7800947
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800065
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7769265
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769265, upper bound: 2.7768811
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800065, upper bound: 2.7768811
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800947, upper bound: 2.7799066
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800654, upper bound: 2.7768811
time: 0.36 seconds

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7795280
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770397, upper bound: 2.7771057
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7799926
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799277, upper bound: 2.7792388
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800039, upper bound: 2.7768811
time: 0.40 seconds

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7801145
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7798526, upper bound: 2.7801145
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800336, upper bound: 2.7799838
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800327, upper bound: 2.7768811
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800401
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.34 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7800401, upper bound: 2.7768811
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800327
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7799838, upper bound: 2.7800336
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7798526
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7771159
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800039
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7792388, upper bound: 2.7799277
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7799926, upper bound: 2.7768811
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7771057, upper bound: 2.7770397
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7795280, upper bound: 2.7768811
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800654
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7799066, upper bound: 2.7800947
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800065
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7769265
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7769265, upper bound: 2.7768811
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7800065, upper bound: 2.7768811
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7800947, upper bound: 2.7799066
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7800654, upper bound: 2.7768811
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7795280
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7770397, upper bound: 2.7771057
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7799926
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7799277, upper bound: 2.7792388
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7800039, upper bound: 2.7768811
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7801145
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7798526, upper bound: 2.7801145
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7800336, upper bound: 2.7799838
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7800327, upper bound: 2.7768811
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800401
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752518, upper bound: 2.7746755
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782794, upper bound: 2.7746755
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789702
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789702
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754041, upper bound: 2.7790491
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7781960, upper bound: 2.7790491
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788146
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7779866, upper bound: 2.7788146
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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782361, upper bound: 2.7746755
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789675
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789675
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788344
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751457, upper bound: 2.7789023
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770375, upper bound: 2.7746755
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786160, upper bound: 2.7746755
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7755174
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750874, upper bound: 2.7753317
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747076, upper bound: 2.7746755
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778045, upper bound: 2.7746755
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7783349
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7753225
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7787676, upper bound: 2.7783621
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7787676, upper bound: 2.7753216
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782929
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7753309
time: 0.33 seconds

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
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746913
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747043
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747043, upper bound: 2.7746755
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746913, upper bound: 2.7746755
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753309, upper bound: 2.7746755
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782929, upper bound: 2.7746755
time: 0.40 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753216, upper bound: 2.7787676
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7783621, upper bound: 2.7787676
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753225, upper bound: 2.7746755
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7783349, upper bound: 2.7746755
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7778045
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747076
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753317, upper bound: 2.7750874
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755174, upper bound: 2.7746755
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7786160
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7770375
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789023, upper bound: 2.7751457
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788344, upper bound: 2.7746755
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789675, upper bound: 2.7746755
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789675, upper bound: 2.7746755
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782361
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788146, upper bound: 2.7779866
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788146, upper bound: 2.7746755
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7790491, upper bound: 2.7781960
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7790491, upper bound: 2.7754041
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789702, upper bound: 2.7746755
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789702, upper bound: 2.7746755
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782794
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7752518
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.36 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7752518, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7782794, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789702
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789702
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7754041, upper bound: 2.7790491
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7781960, upper bound: 2.7790491
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788146
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7779866, upper bound: 2.7788146
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7782361, upper bound: 2.7746755
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789675
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789675
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788344
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7751457, upper bound: 2.7789023
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7770375, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7786160, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7755174
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7750874, upper bound: 2.7753317
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7747076, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7778045, upper bound: 2.7746755
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7783349
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7753225
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7787676, upper bound: 2.7783621
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7787676, upper bound: 2.7753216
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782929
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7753309
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746913
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747043
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7747043, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746913, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7753309, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7782929, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7753216, upper bound: 2.7787676
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7783621, upper bound: 2.7787676
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7753225, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7783349, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7778045
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747076
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7753317, upper bound: 2.7750874
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7755174, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7786160
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7770375
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7789023, upper bound: 2.7751457
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7788344, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7789675, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7789675, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782361
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7788146, upper bound: 2.7779866
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7788146, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7790491, upper bound: 2.7781960
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7790491, upper bound: 2.7754041
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7789702, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7789702, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782794
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7752518
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535127, upper bound: 2.7533155
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535127, upper bound: 2.7533197
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7579100
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7579100
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535131, upper bound: 2.7563966
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535214, upper bound: 2.7564464
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7534652, upper bound: 2.7577225
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7534652, upper bound: 2.7577225
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7549012, upper bound: 2.7564821
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7549012, upper bound: 2.7564821
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533820
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533978
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7542692, upper bound: 2.7533778
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7560946, upper bound: 2.7533778
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533820
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532991, upper bound: 2.7533838
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582459, upper bound: 2.7533354
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582459, upper bound: 2.7533354
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7575138
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7575679
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535250, upper bound: 2.7562011
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535480, upper bound: 2.7563167
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7555630
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7560268
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535687, upper bound: 2.7558570
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535718, upper bound: 2.7560691
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535729
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535729
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535024, upper bound: 2.7535666
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535024, upper bound: 2.7535666
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7565964, upper bound: 2.7532850
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7545022, upper bound: 2.7532625
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536209
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536209
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533188, upper bound: 2.7534824
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533188, upper bound: 2.7534824
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533225, upper bound: 2.7535629
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533225, upper bound: 2.7535629
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7562925, upper bound: 2.7533260
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7548464, upper bound: 2.7533172
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7582293
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7582293
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7539970
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533279, upper bound: 2.7575015
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532883, upper bound: 2.7566529
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533996, upper bound: 2.7539491
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7577203
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7577203
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541317
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541317
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541962
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541962
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536538
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7536538, upper bound: 2.7532625
time: 0.39 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.52 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7535127, upper bound: 2.7533155
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7535127, upper bound: 2.7533197
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7579100
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7579100
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7535131, upper bound: 2.7563966
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7535214, upper bound: 2.7564464
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7534652, upper bound: 2.7577225
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7534652, upper bound: 2.7577225
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7549012, upper bound: 2.7564821
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7549012, upper bound: 2.7564821
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533820
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533978
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7542692, upper bound: 2.7533778
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7560946, upper bound: 2.7533778
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533820
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532991, upper bound: 2.7533838
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7582459, upper bound: 2.7533354
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7582459, upper bound: 2.7533354
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7575138
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7575679
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7535250, upper bound: 2.7562011
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7535480, upper bound: 2.7563167
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7555630
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7560268
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7535687, upper bound: 2.7558570
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7535718, upper bound: 2.7560691
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535729
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535729
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7535024, upper bound: 2.7535666
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7535024, upper bound: 2.7535666
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7565964, upper bound: 2.7532850
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7545022, upper bound: 2.7532625
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536209
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536209
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7533188, upper bound: 2.7534824
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7533188, upper bound: 2.7534824
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7533225, upper bound: 2.7535629
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7533225, upper bound: 2.7535629
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7562925, upper bound: 2.7533260
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7548464, upper bound: 2.7533172
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7582293
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7582293
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7539970
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7533279, upper bound: 2.7575015
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532883, upper bound: 2.7566529
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7533996, upper bound: 2.7539491
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7577203
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7577203
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541317
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541317
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541962
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541962
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536538
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.52
Output dim: 0, lower bound: -2.7536538, upper bound: 2.7532625
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7746913, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7753309, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7782929, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7753216, upper bound: 2.7787676
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7783621, upper bound: 2.7787676
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7753225, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7783349, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7778045
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747076
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7753317, upper bound: 2.7750874
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7755174, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7786160
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7770375
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7789023, upper bound: 2.7751457
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7788344, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7789675, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7789675, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782361
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7788146, upper bound: 2.7779866
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7788146, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7790491, upper bound: 2.7781960
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7790491, upper bound: 2.7754041
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7789702, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7789702, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782794
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7752518
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.52
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
Binary search (step 3): status=Status.UNKNOWN, low=0.1750000, high=0.1875000, mid=0.1875000, abs_max=3.285133123397827
rel_dist={0: [-2.7804845941221, 2.7804845941221004]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.17499998956918716
execution time: 1152.86 seconds
