## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_2.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 817.226686863868


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490)
1: (-233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119)
2: (-244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908)
3: (-388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457)
4: (-395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221)

## BASE Result
execution time: IAR + LP analysis = 1.83 + 1.96 = 3.79 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -817.2521507, upper bound: 817.2521507


# Binary Search by BASE starts (time budget: 1196.21 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=945.3177490234375
rel_dist={0: [-817.2521391840036, 817.2521391840037]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=945.3177490234375
rel_dist={0: [-817.2518432227157, 817.2518432227159]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=945.3177490234375
rel_dist={0: [-817.2513134576823, 817.2513134576823]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=945.3177490234375
rel_dist={0: [-817.2509208504653, 817.2509208504653]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=945.3177490234375
rel_dist={0: [-817.2506632438411, 817.2506632438412]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=945.3177490234375
rel_dist={0: [-817.2505160692408, 817.2505160692408]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=945.3177490234375
rel_dist={0: [-817.2504384746804, 817.2504384746804]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=945.3177490234375
rel_dist={0: [-817.2503978663774, 817.2503978663776]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=945.3177490234375
rel_dist={0: [-817.250377229322, 817.2503772291393]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=945.3177490234375
rel_dist={0: [-817.2503667907724, 817.2503667907724]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=945.3177490234375
rel_dist={0: [-817.2503612532297, 817.2503612532296]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=945.3177490234375
rel_dist={0: [-817.2503583909031, 817.2503583909031]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=945.3177490234375
rel_dist={0: [-817.2503569597413, 817.2503569597206]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=945.3177490234375
rel_dist={0: [-817.2503562443676, 817.2503562441645]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=945.3177490234375
rel_dist={0: [-817.250355886418, 817.2503558864232]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=945.3177490234375
rel_dist={0: [-817.2503557076657, 817.2503557074656]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=945.3177490234375
rel_dist={0: [-817.2503556184678, 817.2503556184677]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=945.3177490234375
rel_dist={0: [-817.250355573426, 817.2503555771218]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=945.3177490234375
rel_dist={0: [-817.2503555583849, 817.2503555551143]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=945.3177490234375
rel_dist={0: [-817.2503556008402, 817.2503555458916]}

## Binary Search Result
Binary search time: 76.15 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1120.06 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110
time: 0.62 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.62 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.62
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.62
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.68 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.76 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.59 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.59
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.59
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.59
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.59
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.73 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.28 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.79 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.87
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.87
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.87
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.87
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.87
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.87
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.87
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.87
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.87
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.87
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.87
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.87
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.87
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.87
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.87
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.87
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
Binary search (step 0): status=Status.VERIFIED, low=0.5000000, high=1.0000000, mid=0.5000000, abs_max=945.3177490234375
rel_dist={0: [-817.2521391840036, 817.2521391840037]}

## Binary search (step 1) starts
Candidate diff: 0.7500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110
time: 0.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.68 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.68
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.68
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.67 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.79 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.32 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.66 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.21 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
Binary search (step 1): status=Status.VERIFIED, low=0.7500000, high=1.0000000, mid=0.7500000, abs_max=945.3177490234375
rel_dist={0: [-817.2521507008507, 817.2521507008505]}

## Binary search (step 2) starts
Candidate diff: 0.8750000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2509110
time: 0.88 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.76
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.76
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2509110

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.64 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.76 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.32 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.69 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.20 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.67 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
Binary search (step 2): status=Status.VERIFIED, low=0.8750000, high=1.0000000, mid=0.8750000, abs_max=945.3177490234375
rel_dist={0: [-817.2521507008508, 817.2521507008505]}

## Binary search (step 3) starts
Candidate diff: 0.9375000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110
time: 0.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.60 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.69 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.84 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.36 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.36
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.36
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.36
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.36
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.74 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.31 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
Binary search (step 3): status=Status.VERIFIED, low=0.9375000, high=1.0000000, mid=0.9375000, abs_max=945.3177490234375
rel_dist={0: [-817.2521507008507, 817.2521507008505]}

## Binary search (step 4) starts
Candidate diff: 0.9687500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110
time: 0.70 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.59 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.74 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.94 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.67 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.67
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.67
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.67
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.67
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.66 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.14 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
Binary search (step 4): status=Status.VERIFIED, low=0.9687500, high=1.0000000, mid=0.9687500, abs_max=945.3177490234375
rel_dist={0: [-817.2521507008507, 817.252150700851]}

## Binary search (step 5) starts
Candidate diff: 0.9843750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110
time: 0.70 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.62 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.69 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.15 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.63 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.14 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.74 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
Binary search (step 5): status=Status.VERIFIED, low=0.9843750, high=1.0000000, mid=0.9843750, abs_max=945.3177490234375
rel_dist={0: [-817.2521507008507, 817.2521507008507]}

## Binary search (step 6) starts
Candidate diff: 0.9921875


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110
time: 0.68 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.58 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.64 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.66 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.11 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.72 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.25 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
Binary search (step 6): status=Status.VERIFIED, low=0.9921875, high=1.0000000, mid=0.9921875, abs_max=945.3177490234375
rel_dist={0: [-817.2521507008507, 817.2521507008505]}

## Binary search (step 7) starts
Candidate diff: 0.9960938


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110
time: 0.68 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.60 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.68 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.62 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.13 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.71 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.33 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
Binary search (step 7): status=Status.VERIFIED, low=0.9960938, high=1.0000000, mid=0.9960938, abs_max=945.3177490234375
rel_dist={0: [-817.2521507008507, 817.2521507008507]}

## Binary search (step 8) starts
Candidate diff: 0.9980469


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110
time: 0.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.65 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.65 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.12 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.12
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.12
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.12
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.12
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.67 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.25 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.76 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
Binary search (step 8): status=Status.VERIFIED, low=0.9980469, high=1.0000000, mid=0.9980469, abs_max=945.3177490234375
rel_dist={0: [-817.2521507008507, 817.2521507008505]}

## Binary search (step 9) starts
Candidate diff: 0.9990234


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110
time: 0.66 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.51 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.66 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.83 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.34 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.67 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.15 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
Binary search (step 9): status=Status.VERIFIED, low=0.9990234, high=1.0000000, mid=0.9990234, abs_max=945.3177490234375
rel_dist={0: [-817.2521507008507, 817.2521507008505]}

## Binary search (step 10) starts
Candidate diff: 0.9995117


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110
time: 0.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.61 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.61
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.61
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.63 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.63 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.11 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.71 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.24 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.24
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.73 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.34 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
Binary search (step 10): status=Status.VERIFIED, low=0.9995117, high=1.0000000, mid=0.9995117, abs_max=945.3177490234375
rel_dist={0: [-817.2521507008507, 817.2521507008505]}

## Binary search (step 11) starts
Candidate diff: 0.9997559


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110
time: 0.65 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.47 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.47
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.47
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.62 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.67 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.18 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.18
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.18
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.18
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.18
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.64 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.64 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
Binary search (step 11): status=Status.VERIFIED, low=0.9997559, high=1.0000000, mid=0.9997559, abs_max=945.3177490234375
rel_dist={0: [-817.2521507008507, 817.2521507008505]}

## Binary search (step 12) starts
Candidate diff: 0.9998779


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110
time: 0.66 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.58 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.62 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.73 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.28 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.65 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.21 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.26
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
Binary search (step 12): status=Status.VERIFIED, low=0.9998779, high=1.0000000, mid=0.9998779, abs_max=945.3177490234375
rel_dist={0: [-817.2521507008507, 817.2521507008507]}

## Binary search (step 13) starts
Candidate diff: 0.9999390


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110
time: 0.63 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.51 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.69 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.68 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.25 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.25
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.25
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.25
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.25
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.67 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.20 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
Binary search (step 13): status=Status.VERIFIED, low=0.9999390, high=1.0000000, mid=0.9999390, abs_max=945.3177490234375
rel_dist={0: [-817.2521507008507, 817.2521507008505]}

## Binary search (step 14) starts
Candidate diff: 0.9999695


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110
time: 0.66 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.52 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.66 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.74 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.38 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.65 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.74 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
Binary search (step 14): status=Status.VERIFIED, low=0.9999695, high=1.0000000, mid=0.9999695, abs_max=945.3177490234375
rel_dist={0: [-817.2521507008507, 817.2521507008507]}

## Binary search (step 15) starts
Candidate diff: 0.9999847


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110
time: 0.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.62 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.62
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.62
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.69 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.68 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.18 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.18
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.18
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.18
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.18
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.69 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.27 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.78 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.41
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.41
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.41
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.41
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.41
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.41
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.41
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.41
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.41
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.41
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.41
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.41
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.41
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.41
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.41
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.41
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
Binary search (step 15): status=Status.VERIFIED, low=0.9999847, high=1.0000000, mid=0.9999847, abs_max=945.3177490234375
rel_dist={0: [-817.2521507008507, 817.252150700851]}

## Binary search (step 16) starts
Candidate diff: 0.9999924


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110
time: 0.70 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.59 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.69 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.70 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.27 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.27
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.27
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.27
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.27
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.68 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.19 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.34 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.34
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
Binary search (step 16): status=Status.VERIFIED, low=0.9999924, high=1.0000000, mid=0.9999924, abs_max=945.3177490234375
rel_dist={0: [-817.2521507008507, 817.2521507008505]}

## Binary search (step 17) starts
Candidate diff: 0.9999962


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110
time: 0.67 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.52 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.68 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.75 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.28 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.66 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.25 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
Binary search (step 17): status=Status.VERIFIED, low=0.9999962, high=1.0000000, mid=0.9999962, abs_max=945.3177490234375
rel_dist={0: [-817.2521507008507, 817.2521507008505]}

## Binary search (step 18) starts
Candidate diff: 0.9999981


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110
time: 0.72 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 0, lower bound: -817.2518496, upper bound: 817.2509110

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.66 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.65 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.11 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.66 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.21 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.71 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
Binary search (step 18): status=Status.VERIFIED, low=0.9999981, high=1.0000000, mid=0.9999981, abs_max=945.3177490234375
rel_dist={0: [-817.2521507008507, 817.2521507008505]}

## Binary search (step 19) starts
Candidate diff: 0.9999990


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2509110
time: 0.82 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2518496
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 0, lower bound: -817.2509110, upper bound: 817.2509110

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
time: 0.66 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
time: 0.65 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.21 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.21
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.21
Output dim: 0, lower bound: -817.2504614, upper bound: 817.2504879
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.21
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.21
Output dim: 0, lower bound: -817.2504879, upper bound: 817.2504614

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
time: 0.62 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.13 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2501029
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -817.2501029, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -817.2500421, upper bound: 817.2500421

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490
1: -233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119
2: -244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908
3: -388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457
4: -395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
time: 0.75 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -817.1596005, upper bound: 817.1596005
Binary search (step 19): status=Status.VERIFIED, low=0.9999990, high=1.0000000, mid=0.9999990, abs_max=945.3177490234375
rel_dist={0: [-817.2521507008507, 817.2521507008507]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.9999990463256836
execution time: 1039.53 seconds
