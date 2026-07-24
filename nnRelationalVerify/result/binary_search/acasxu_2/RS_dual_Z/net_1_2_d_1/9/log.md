## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_2.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 57.280903066


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068)
1: (-16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383)
2: (-16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280)
3: (-27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894)
4: (-25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331)

## BASE Result
execution time: IAR + LP analysis = 2.09 + 1.59 = 3.68 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -57.5687468, upper bound: 57.5687468


# Binary Search by BASE starts (time budget: 1196.32 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=66.57380676269531
rel_dist={0: [-57.5687467976788, 57.5687467976788]}

## Binary search (step 1) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=66.57380676269531
rel_dist={0: [-57.5687467976788, 57.5687467976788]}

## Binary search (step 2) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=66.57380676269531
rel_dist={0: [-57.5686962838552, 57.5686962838552]}

## Binary search (step 3) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=66.57380676269531
rel_dist={0: [-57.56848922332033, 57.56848922332034]}

## Binary search (step 4) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=66.57380676269531
rel_dist={0: [-57.568351401632306, 57.5683514016323]}

## Binary search (step 5) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=66.57380676269531
rel_dist={0: [-57.568257500820835, 57.568257500820835]}

## Binary search (step 6) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=66.57380676269531
rel_dist={0: [-57.5682064165504, 57.56820641655041]}

## Binary search (step 7) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=66.57380676269531
rel_dist={0: [-57.568180214636605, 57.56818021463661]}

## Binary search (step 8) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=66.57380676269531
rel_dist={0: [-57.56816602745435, 57.56816602745435]}

## Binary search (step 9) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=66.57380676269531
rel_dist={0: [-57.568158614525274, 57.56815861452529]}

## Binary search (step 10) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=66.57380676269531
rel_dist={0: [-57.56815490806733, 57.56815490806733]}

## Binary search (step 11) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=66.57380676269531
rel_dist={0: [-57.56815305485149, 57.56815305485149]}

## Binary search (step 12) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=66.57380676269531
rel_dist={0: [-57.568152128269546, 57.568152128269546]}

## Binary search (step 13) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=66.57380676269531
rel_dist={0: [-57.5681516650294, 57.568151665029404]}

## Binary search (step 14) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=66.57380676269531
rel_dist={0: [-57.568151433506756, 57.568151433506756]}

## Binary search (step 15) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=66.57380676269531
rel_dist={0: [-57.56815131792488, 57.56815131792487]}

## Binary search (step 16) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=66.57380676269531
rel_dist={0: [-57.56815126044128, 57.56815126044128]}

## Binary search (step 17) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=66.57380676269531
rel_dist={0: [-57.56815123432093, 57.56815126003005]}

## Binary search (step 18) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=66.57380676269531
rel_dist={0: [-57.568151283478784, 57.568151286271956]}

## Binary Search Result
Binary search time: 71.11 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1125.21 seconds

## Binary search (step 0) starts
Candidate diff: 0.2500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5684507, upper bound: 57.5684509
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5684509, upper bound: 57.5684507
time: 0.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.46 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 0, lower bound: -57.5684507, upper bound: 57.5684509
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 0, lower bound: -57.5684509, upper bound: 57.5684507

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.51 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.60 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.29 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.57 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.25 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250957, upper bound: 57.4243014
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250957, upper bound: 57.4243015
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250957, upper bound: 57.4243014
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250957, upper bound: 57.4243015
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250957, upper bound: 57.4243014
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250957, upper bound: 57.4243014
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250957, upper bound: 57.4243015
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
time: 0.58 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.33 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.4250957, upper bound: 57.4243014
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.4250957, upper bound: 57.4243015
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.4250957, upper bound: 57.4243014
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.4250957, upper bound: 57.4243015
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.4250957, upper bound: 57.4243014
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.4250957, upper bound: 57.4243014
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.4250957, upper bound: 57.4243015
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250957, upper bound: 57.4243014
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250957, upper bound: 57.4243014
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250957, upper bound: 57.4243014
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250957, upper bound: 57.4243014
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.61 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4250957, upper bound: 57.4243014
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4250957, upper bound: 57.4243014
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4250957, upper bound: 57.4243014
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4250957, upper bound: 57.4243014
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250957
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.53 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.44 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.37 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=66.57380676269531
rel_dist={0: [-57.5687467976788, 57.5687467976788]}

## Binary search (step 1) starts
Candidate diff: 0.1250000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5684507, upper bound: 57.5684509
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5684509, upper bound: 57.5684507
time: 0.53 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.26 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.26
Output dim: 0, lower bound: -57.5684507, upper bound: 57.5684509
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.26
Output dim: 0, lower bound: -57.5684509, upper bound: 57.5684507

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.59 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.55 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.24 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.53 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.34 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250379, upper bound: 57.4243014
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250379, upper bound: 57.4243014
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250379, upper bound: 57.4243014
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250379, upper bound: 57.4243014
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250379, upper bound: 57.4243014
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250379, upper bound: 57.4243014
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
time: 0.57 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.30 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4250379, upper bound: 57.4243014
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4250379, upper bound: 57.4243014
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4250379, upper bound: 57.4243014
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4250379, upper bound: 57.4243014
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4250379, upper bound: 57.4243014
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4250379, upper bound: 57.4243014
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250379, upper bound: 57.4243014
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250379, upper bound: 57.4243014
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.61 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.43 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4250379, upper bound: 57.4243014
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4250379, upper bound: 57.4243014
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4250379
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.83 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 1.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 1.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.40 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=66.57380676269531
rel_dist={0: [-57.5687467976788, 57.5687467976788]}

## Binary search (step 2) starts
Candidate diff: 0.0625000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5683992, upper bound: 57.5683992
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5683992, upper bound: 57.5683992
time: 0.53 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.33 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.33
Output dim: 0, lower bound: -57.5683992, upper bound: 57.5683992
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.33
Output dim: 0, lower bound: -57.5683992, upper bound: 57.5683992

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.57 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.52 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.29 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.57 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.55 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4247286, upper bound: 57.4243014
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4247286, upper bound: 57.4243014
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243014, upper bound: 57.4243014
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4247286, upper bound: 57.4243014
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
time: 0.59 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -57.4247286, upper bound: 57.4243014
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -57.4247286, upper bound: 57.4243014
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -57.4243014, upper bound: 57.4243014
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -57.4247286, upper bound: 57.4243014
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.31
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4247286, upper bound: 57.4243014
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243014, upper bound: 57.4243015
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4247286, upper bound: 57.4243015
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243014, upper bound: 57.4243015
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243014, upper bound: 57.4243015
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 3.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243014, upper bound: 57.4247286
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243014, upper bound: 57.4243015
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.71 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.57 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4247286, upper bound: 57.4243014
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243014, upper bound: 57.4243015
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4247286, upper bound: 57.4243015
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243014, upper bound: 57.4243015
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243014, upper bound: 57.4243015
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243014, upper bound: 57.4247286
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243014, upper bound: 57.4243015
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4247286
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.57
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
time: 0.56 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -57.4037525, upper bound: 57.4037525

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961705, upper bound: 57.3961705
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 2.21 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=66.57380676269531
rel_dist={0: [-57.5686962838552, 57.5686962838552]}

## Binary Search with RS_dual_Z Result
status: None
Maximum delta epsilon: None
execution time: 1126.57 seconds
